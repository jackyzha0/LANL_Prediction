import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import numpy as np
import os

train_dir_path = "data/"
filename = train_dir_path + "train.csv"
filenames = [filename]
record_defaults = [tf.float32] * len(filenames)  # Only provide defaults for the selected columns
x_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[0])
y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1])
dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
fix_len = 150000
x_list = range(fix_len)

batched_dataset = dataset.batch(fix_len*2)
iterator = batched_dataset.make_one_shot_iterator()
random_ind = np.random.randint(0,fix_len)
next_element = iterator.get_next()

def get_example():
    return np.array(sess.run(next_element))[:,0,random_ind:random_ind+fix_len]

def get_batch(batchsize, shuffle=True):
    batch = []
    for i in range(batchsize):
        batch.append(get_example())
    if shuffle:
        np.random.shuffle(batch)
    return batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    inputs = tf.placeholder(tf.float32, [None, None, num_mfccs*2])

    ex = get_example()
    print(ex)
