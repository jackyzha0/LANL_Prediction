import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
config = tf.ConfigProto()

learning_rate = 1e-3
batchsize = 4
epochs = 10

#LSTM Params
num_hidden = 50
num_classes = 10

train_dir_path = "data/"
filename = train_dir_path + "train.csv"
dbsize = 6000000
fix_len = 150000
filenames = [filename]

record_defaults = [tf.float32] * len(filenames)  # Only provide defaults for the selected columns
x_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[0])
y_dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[1])
dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
x_list = range(fix_len)

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

graph = tf.Graph()
with graph.as_default():

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

    #Define network architecture
    x_acoustic = tf.placeholder(tf.float32, [None, fix_len])
    y_timetofailure = tf.placeholder(tf.float32, [None, 1])
    x_reshape = tf.reshape(x_acoustic, [-1, fix_len, 1])
    # y_timetofailure = tf.expand_dims(y_timetofailure, -1)
    # print(y_timetofailure.get_shape())

    def RNN(x):

        weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, 268, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    net = tf.layers.conv1d(x_reshape, 16, 10000, strides = 3) #Form [input layer, #filters, kernel_size]
    print(net.get_shape())
    net = tf.layers.conv1d(net, 8, 5000, strides = 5)
    print(net.get_shape())
    net = tf.layers.conv1d(net, 4, 1000, strides = 2)
    print(net.get_shape())
    net = tf.layers.conv1d(net, 2, 500, strides = 5)
    print(net.get_shape())
    net = tf.layers.conv1d(net, 1, 100, strides = 2)
    print(net.get_shape())

    # net = RNN(net)
    rnet = tf.reshape(net, [-1, 268])
    print(rnet.get_shape())

    logits = tf.contrib.layers.fully_connected(rnet, 1)
    print(logits.get_shape())


    loss = tf.losses.mean_squared_error(y_timetofailure, logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session(graph=graph, config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for epoch in range(epochs):
        for batch in range(int(dbsize / (fix_len * batchsize))):

            ex_batch = get_batch(batchsize)
            # print(ex_batch)
            x,y = np.split(ex_batch, 2, axis = 1)
            x = np.squeeze(x)
            y = np.squeeze(y)[:,-1:]

            out = sess.run([optimizer, loss], feed_dict={x_acoustic: x, y_timetofailure: y})
            print(out)
            # out = sess.run([train_op, loss, logits],
            # feed_dict={x_acoustic: x, y_timetofailure: y})

            # print('Pred: ' + str(np.array(out)[2][-1]))
            # print('Actual: ' + str(ex_batch[1][-1]))
            # print('Loss: ' + str(np.array(out)[1]))
            # print('______')
