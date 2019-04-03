import csvtools
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set Tensorflow run options
run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
config = tf.ConfigProto()

### PARAMETERS ###
batchsize = 32
fix_len = 150000
learning_rate = 1e-5
iters = 1000
dbsize = 6000000

num_hidden = 50
num_classes = 10

def plot(x):
    fig, ax1 = plt.subplots()
    t = np.arange(len(x[0]))

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('predicted time', color=color)
    ax1.plot(t, x[0], color=color)
    color = 'tab:blue'
    ax1.plot(t, x[1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('time until fault', color=color)
    # ax2.plot(t, x[1], color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

graph = tf.Graph()
with graph.as_default():

    def RNN(x):

        weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, 140, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    x_acoustic = tf.placeholder(tf.float32, [None, fix_len])
    y_timetofailure = tf.placeholder(tf.float32, [None, 1])
    x_reshape = tf.reshape(x_acoustic, [-1, fix_len, 1])

    net = tf.layers.conv1d(x_reshape, 64, 100, strides = 10, kernel_initializer=tf.contrib.layers.xavier_initializer()) #Form [input layer, #filters, kernel_size]
    print(net.get_shape())
    net = tf.layers.conv1d(net, 32, 100, strides = 10, kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(net.get_shape())
    net = tf.layers.conv1d(net, 16, 100, strides = 10, kernel_initializer=tf.contrib.layers.xavier_initializer())
    print(net.get_shape())

    f_net = tf.contrib.layers.flatten(net)
    print(f_net.get_shape())

    r_net = RNN(net)

    logits = tf.layers.dense(f_net, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
    print('_____')
    print(logits.get_shape())
    print(y_timetofailure.get_shape())

    loss = tf.reduce_mean(tf.losses.mean_squared_error(logits, y_timetofailure))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1e-8).minimize(loss)

with tf.Session(graph=graph, config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for _ in range(iters):
        batch = csvtools.get_batch(batchsize)
        x,y = np.split(batch, 2, axis = 1)
        x = np.squeeze(x, axis=(1,))
        y = np.squeeze(y, axis=(1,))[:,-1:]

        # print('Running for logits...',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        out = sess.run([optimizer, loss, logits], feed_dict={x_acoustic: x, y_timetofailure: y})
        if (_ % 25 == 0):
            plot([out[2],y])
        # print('Pred: ' + str(np.array(out)[2]))
        # print('Actual: ' + str(y))
        print('Batch ' + str(_) + ' | Loss: ' + str(np.array(out)[1]))
        # print('______')
