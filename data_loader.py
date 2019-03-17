import tensorflow as tf
# tf.enable_eager_execution()

FILENAME = 'data.tfrecords'

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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batched_dataset = dataset.batch(fix_len*2)
    iterator = batched_dataset.make_one_shot_iterator()
    random_ind = np.random.randint(0,fix_len)
    next_element = iterator.get_next()

    print(np.array(sess.run(next_element))[:,0,random_ind:random_ind+fix_len])
    print(len(np.array(sess.run(next_element))[:,0,random_ind:random_ind+fix_len][0]))
