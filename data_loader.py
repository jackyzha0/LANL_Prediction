#Init
import tensorflow as tf
import os

train_dir_path = "data/"
filename = train_dir_path + "train.csv"

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            acoustic_data,time_to_failure = line.strip().split(",")
            # Run the Print ob
            print(acoustic_data,time_to_failure)
