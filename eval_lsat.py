import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import vgg_preprocessing
from vgg import vgg_19, vgg_arg_scope
import time
import os
from train_lsat import get_split, load_batch
import matplotlib.pyplot as plt
from cam_utils import *
plt.style.use('ggplot')
slim = tf.contrib.slim

#State your log directory where you can retrieve your model
log_dir = './log'

#Create a new evaluation log directory to visualize the validation process
log_eval = './log_eval_test'

#State the dataset directory where the validation set is found
dataset_dir = '/home/elebouder/Data/landsat/tfrecords/'

#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 8

#State the number of epochs to evaluate
num_epochs = 1

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)


def run():
    #Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('validation', dataset_dir)
        images, raw_images, labels = load_batch(dataset, batch_size = batch_size, is_training = False)
        imagescam, rcam, lcam = load_batch(dataset, batch_size=batch_size, is_training=False, cam=True)
        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        #placedholders for CAM compute
        y_ = tf.placeholder(tf.int64, [None])
        x = tf.placeholder_with_default(images, (None, 224, 224, 3))


        

        #Now create the inference model but set is_training=False
        with slim.arg_scope(vgg_arg_scope()):
            logits, end_points = vgg_19(x, num_classes = dataset.num_classes, is_training = False, global_pool=True)

        
        #Get the class activation maps
        class_activation_map = get_class_map(1, end_points['vgg_19/conv6'], 224)
        
         
       
        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
        
        #Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['vgg_19/fc8'], 1)
        # accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        accuracy, accuracy_update = tf.metrics.accuracy(labels, predictions)
        metrics_op = tf.group(accuracy_update)

        #Create the global step and an increment op for monitoring
        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        
        #placedholders for CAM compute
        y = logits

        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            print 'start', start_time
            _ = sess.run(metrics_op) 
            global_step_count = sess.run(global_step_op)
            accuracy_value = sess.run(accuracy)
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
            
            print 'starting cam inspection'
            #produce and save CAMs every 10 steps
            inspect_class_activation_map(sess, class_activation_map, end_points['vgg_19/conv6'], imagescam, lcam, global_step_count, batch_size, x, y_, y)
             
            print 'ending cam inspection'
            return accuracy_value


        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()
        

        #Get your supervisor
        #sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)
        
        #global_step_tensor = tf.Variable(7530, trainable=False, name='global_step')        
        #Now we are ready to run in one session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restore_fn(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
           
                #from tensorflow.python import debug as tfdb
                #sess = tfdb.LocalCLIDebugWrapperSession(sess)
                #tf.train.global_step(sess, global_step_tensor)
                for step in xrange(num_steps_per_epoch * num_epochs):                      
                    print global_step
                    sess.run(global_step)
                 
                    #print vital information every start of the epoch as always
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                        logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    
                    #Compute summaries every 10 steps and continue evaluating
                    if step % 10 == 0:
                        print 'mod 10'
                        eval_step(sess, metrics_op = metrics_op, global_step = global_step)
                        summaries = sess.run(my_summary_op)
                        #sv.summary_computed(sess, summaries)
                            

                    #Otherwise just run as per normal
                    else:
                        print 'next step'
                        eval_step(sess, metrics_op = metrics_op, global_step = global_step)
            

            coord.request_stop()
            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions = sess.run([raw_images, labels, predictions])
            for i in range(10):
                image, label, prediction = raw_images[i], labels[i], predictions[i]
                prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
                text = 'Prediction: %s \n Ground Truth: %s' %(prediction_name, label_name)
                img_plot = plt.imshow(image)

                #Set up the plot and hide axes
                plt.title(text)
                img_plot.axes.get_yaxis().set_ticks([])
                img_plot.axes.get_xaxis().set_ticks([])
                plt.show()
            coord.request_stop()
            coord.join(threads)
            
            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')

if __name__ == '__main__':
    run()
