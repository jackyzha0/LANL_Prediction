import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
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
fix_len = 1500000
x_list = range(fix_len)

def smooth(x):
    n = 50  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b,a,x)
    return yy

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batched_dataset = dataset.batch(fix_len*2)
    iterator = batched_dataset.make_one_shot_iterator()
    random_ind = np.random.randint(0,fix_len)
    next_element = iterator.get_next()

    def get_example():
        return np.array(sess.run(next_element))[:,0,random_ind:random_ind+fix_len]

    ex = get_example()
    ex = get_example()

    fig, ax1 = plt.subplots()
    t = np.arange(fix_len)

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('acoustic data', color=color)
    ax1.plot(t, ex[0], color='tab:green')
    ax1.plot(t, smooth(ex[0]), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('time until fault', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, smooth(ex[1]), color=color)
    ax2.plot(t, ex[1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
