import os
from shutil import copyfile



parent_dir = "/home/elebouder/Data/landsat/"
src_dir = parent_dir + "training_clips/"
"""Make sure the split data dirs don't exist before running this code"""
root_dir = src_dir + "split/"
next_dir = root_dir + "photos/"
pos_dir = next_dir + "pos/"
neg_dir = next_dir + "neg/"
train_src_dir = src_dir + "Train/"
test_src_dir = src_dir + "Test/"



os.mkdir(root_dir)
os.mkdir(next_dir)
os.mkdir(pos_dir)
os.mkdir(neg_dir)


trainf = open(parent_dir + "train_llbl.txt", 'r')
for line in trainf:
    tp = line.split(" ")
    name = tp[0]
    class_id = int(tp[1])
    if class_id == 1:
        src = train_src_dir + name
        dst = pos_dir + name
    elif class_id == 0:
        src = train_src_dir + name
        dst = neg_dir + name
    copyfile(src, dst)
trainf.close()

testf = open(parent_dir + "test_llbl.txt", 'r')
for line in testf:
    tp = line.split(" ")
    name = tp[0]
    class_id = int(tp[1])
    if class_id == 1:
        src = test_src_dir + name
        dst = pos_dir + name
    elif class_id == 0:
        src = test_src_dir + name
        dst = neg_dir + name
    copyfile(src, dst)
trainf.close()



