import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import numpy as np
import os

learning_rate = 1e-3
batchsize = 16
train_dir_path = "data/"
filename = train_dir_path + "train.csv"
fix_len = 150000
filenames = [filename]

record_defaults = [tf.float32] * len(filenames)  # Only provide defaults for the selected columns
x_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[0])
y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1])
dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
x_list = range(fix_len)

batched_dataset = dataset.batch(fix_len*2)
iterator = batched_dataset.make_one_shot_iterator()
random_ind = np.random.randint(0,fix_len)
next_element = iterator.get_next()

def get_example():
    return np.array(sess.run(next_element))[:,0,random_ind:random_ind+fix_len]

def get_batch(batchsize, shuffle=True):
    batch = np.zeros(shape=(batchsize,2,fix_len))
    for i in range(batchsize):
        batch[i] = get_example()
    if shuffle:
        np.random.shuffle(batch)
    return batch

def plot(x):
    fig, ax1 = plt.subplots()
    t = np.arange(len(x[0]))

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('acoustic data', color=color)
    ax1.plot(t, x[0], color='tab:red')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('time until fault', color=color)
    ax2.plot(t, x[1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_acoustic = tf.placeholder(tf.float32, [None, fix_len])
    y_timetofailure = tf.placeholder(tf.float32, [None, fix_len])

    x = get_batch(batchsize)
