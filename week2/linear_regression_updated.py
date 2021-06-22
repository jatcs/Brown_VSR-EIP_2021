# changed implementation to properly implement matrix multiplication of layers

import numpy as np
import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from IPython import embed
import matplotlib.pyplot as plt


# initialization for the weights / biases (from Shengze's functionApprox-0619.py)
def parameters_initialization(size):
    return tf.Variable(tf.random_normal(shape=size, mean=0., stddev=0.1, dtype=tf.float64), dtype=tf.float64)


# basically DNN from shengze's implementation but with comments to check that I understand
def network(input_, weights, biases):
    activation = input_
    n_layers = len(weights)
    for i in range(n_layers - 1):
        # use tanh so activation stays within -1 and 1?
        activation = tf.tanh(tf.add(tf.matmul(activation, weights[i]), biases[i]))
        # ^ same as doing W (a) + b for each node, but on a larger scale

    # not using tanh for last layer so its range isn't diminished
    return tf.add(tf.matmul(activation, weights[n_layers - 1]), biases[n_layers - 1])


if __name__ == '__main__':

    # range = np.random
    # random batch of data from
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    # allow these batches to be compatible with tf.matmul (have it be rank 2 (number of dimensions I suppose))
    train_X = train_X.reshape(-1, 1)
    train_Y = train_Y.reshape(-1, 1)

    num_samples = train_X.shape
    # unseen X for the target to guess
    random_X = (max(train_X) - min(train_X)) * np.random.random_sample(num_samples,) + min(train_X)

    # details for model size
    num_inputs = 1
    num_outputs = 1
    num_layers = 1
    num_nodes = 3

    x_tf = tf.placeholder(dtype=tf.float64, shape=[None, 1])
    y_tf = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    # updated to use lists instead of one variable with shape=(num_layers, num_nodes)
    model_shape = [num_inputs] + num_layers * [num_nodes] + [num_outputs]
    W = [parameters_initialization([model_shape[i - 1], model_shape[i]]) for i in range(1, len(model_shape))]
    b = [parameters_initialization([1, model_shape[i]]) for i in range(1, len(model_shape))]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    y_pred_tf = network(train_X, W, b)
    loss = tf.reduce_mean(tf.square(y_pred_tf - y_tf))

    learning_rate = 1.0e-1
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # desired accuracy
    epsilon = 0.00001
    # max iteration
    num_iterations = 100
    
    # train
    for i in range(num_iterations):
        print('\n' + ("=" * 30) + " iteration {} ".format(i + 1) + ("=" * 30))
    
        # dictionary feeding to the graph
        tf_dict = {x_tf: train_X, y_tf: train_Y}
    
        # Loading the previous parameters from graph
        w_prev, b_prev = sess.run([W, b])
        # Computing loss from graph (input and output data required)
        prev_loss = sess.run(loss, feed_dict=tf_dict)

        sess.run(train, feed_dict=tf_dict)
        #^ FailedPreconditionError: Attempting to use uninitialized value beta1_power [[{{node beta1_power/read}}]]
    
        # check that its approaching 0 (where min occurs)
        current_loss = sess.run(loss, feed_dict=tf_dict)
        current_loss_diff = abs(current_loss - prev_loss)
        print('current loss difference =', current_loss_diff)
        if current_loss_diff < epsilon:
            print("\nDesired accuracy reached !! Woop!")
            break
    
    
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    plt.plot(train_X, y_pred_tf, 'green', label='Approximation')
    
    sess.close()
