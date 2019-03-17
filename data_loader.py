#Init
import tensorflow as tf
import pandas as pd
tf.enable_eager_execution()

FILENAME = 'data.tfrecords'

import numpy as np

import os

train_dir_path = "data/"
filename = train_dir_path + "train.csv"
chunksize = 10 ** 8

acoustic_data = []
time_to_failure = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        csv = chunk.values
        with tf.python_io.TFRecordWriter(FILENAME) as writer:
            for row in csv:
                features, label = row[:-1], row[-1]
                example = tf.train.Example()
                example.features.feature["features"].float_list.value.extend(features)
                example.features.feature["label"].float_list.value.append(label)
                writer.write(example.SerializeToString())

#150k lines of input
