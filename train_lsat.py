import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import vgg_preprocessing
import lenet_preprocessing
from vgg import vgg_19, vgg_arg_scope
import os
import time
slim = tf.contrib.slim
import numpy as np
#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
dataset_dir = '/home/elebouder/Data/landsat/tfrecords/'

#State where your log file is at. If it doesn't exist, create it.
log_dir = './log'

#State where your checkpoint file is
checkpoint_file = './vgg_19.ckpt'

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224

#State the number of classes to predict:
num_classes = 2

#create the label/class mapping
labels_to_name = {1: 'pos', 0: 'neg'}

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'lsat_%s_*.tfrecord'

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured chest xray image image that is either pos or neg.',
    'label': 'A label that is as such -- 0:neg, 1:pos'
}


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 30

#State your batch size
batch_size = 32

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

#============== DATASET LOADING ======================
#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir='/home/elebouder/Data/landsat/tfrecords/', file_pattern=file_pattern, file_pattern_for_counting='lsat'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True, cam=False):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    if not cam:
        image = vgg_preprocessing.preprocess_image(raw_image, height, width, is_training)
    else:
        image = lenet_preprocessing.preprocess_image(raw_image, height, width, is_training)
    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels

def run():
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        #First create the dataset and load one batch
        dataset = get_split('train', dataset_dir, file_pattern=file_pattern)
        images, _, labels = load_batch(dataset, batch_size=batch_size)

        #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        with slim.arg_scope(vgg_arg_scope()):
            logits, end_points = vgg_19(images, num_classes = dataset.num_classes, is_training = True, global_pool=True)

        #Define the scopes that you want to exclude for restoration
        exclude = ['vgg_19/fc8', 'vgg_19/fc7', 'vgg_19/fc6', 'vgg_19/conv6']
        # edit above and below lines based on variables to restore and checkpoint type
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        print end_points
        predictions = tf.argmax(end_points['vgg_19/fc8'], 1)
        probabilities = end_points['vgg_19/fc8']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)


        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()
        

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        print variables_to_restore
        # Uncomment the following block of code to use a .chkpt formatted checkpoint
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
        
        

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        #sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)


        #Run the managed session
        #with sv.managed_session() as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restore_fn(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not coord.should_stop():
 
                for step in xrange(num_steps_per_epoch * num_epochs):
                    #At the start of every epoch, show the vital information:
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                        learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                        logging.info('Current Learning Rate: %s', learning_rate_value)
                        logging.info('Current Streaming Accuracy: %s', accuracy_value)

                        # optionally, print your logits and predictions for a sanity check that things are going fine.
                        logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                        print 'logits: \n', logits_value
                        print 'Probabilities: \n', probabilities_value
                        print 'predictions: \n', predictions_value
                        print 'Labels:\n:', labels_value

                    #Log the summaries every 10 step.
                    if step % 10 == 0:
                        loss, _ = train_step(sess, train_op, sv.global_step)
                        summaries = sess.run(my_summary_op)
                        sv.summary_computed(sess, summaries)
                    
                    #If not, simply run the training step
                    else:
                        loss, _ = train_step(sess, train_op, sv.global_step)

                #We log the final training loss and accuracy
                logging.info('Final Loss: %s', loss)
                logging.info('Final Accuracy: %s', sess.run(accuracy))

                #Once all the training has been done, save the log files and checkpoint model
                logging.info('Finished training! Saving model to disk now.')
                # saver.save(sess, "./flowers_model.ckpt")
                sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    run()


