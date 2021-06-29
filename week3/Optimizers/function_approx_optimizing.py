# week 2/3 assignment - test out effectiveness of different parameters and optimizers
# Jessica Turner
import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from IPython import embed


def function_target(x):
    return (5 + np.sin(x) + np.sin(2 * x) + np.sin(3 * x) + np.sin(4 * x)) \
        if (x < 0) else np.cos(10 * x)


def parameters_initialization(size_list):
    # function for initializing weights and biases with random values
    return tf.Variable(tf.random_normal(shape=size_list, mean=0., stddev=0.1, dtype=tf.float64), dtype=tf.float64)


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

    # ======================================
    #   Saving settings
    # ======================================
    current_directory = os.getcwd()
    results_dir = "/Output/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)

    plots_dir = "/Plots/"
    save_plots_to = current_directory + plots_dir
    if not os.path.exists(save_plots_to):
        os.makedirs(save_plots_to)


    # constants
    X_DIM, LOWER_BOUND = 0, 0
    Y_DIM, UPPER_BOUND = 1, 1
    # initialize training set
    num_points = 100
    domain = (-1 * math.pi, math.pi)
    train_X = np.asarray(((domain[UPPER_BOUND] - domain[LOWER_BOUND]) * np.random.random_sample(num_points)) + domain[LOWER_BOUND])
    train_Y = np.asarray([function_target(x) for x in train_X])

    train_X = train_X.reshape(-1, 1)
    train_Y = train_Y.reshape(-1, 1)

    # details for model size
    num_inputs = 1
    num_outputs = 1
    num_layers = 2
    num_nodes = 75

    # input and output nodes
    x_tf = tf.placeholder(dtype=tf.float64, name="X", shape=[None, 1])
    y_tf = tf.placeholder(dtype=tf.float64, name="Y", shape=[None, 1])

    # parameters for middle nodes
    # updated to use lists instead of one variable with shape=(num_layers, num_nodes)
    model_shape = [num_inputs] + num_layers * [num_nodes] + [num_outputs]
    W = [parameters_initialization([model_shape[i - 1], model_shape[i]]) for i in range(1, len(model_shape))]
    b = [parameters_initialization([1, model_shape[i]]) for i in range(1, len(model_shape))]

    # node operations
    y_pred_tf = network(x_tf, W, b)
    loss = tf.reduce_mean(tf.square(y_pred_tf - y_tf))

    learning_rate = 1e-1
    train = tf.train.MomentumOptimizer(learning_rate, momentum=0.0625).minimize(loss)

    # must be after graph initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # desired accuracy
    epsilon = 0.00000001
    # max iteration
    num_iterations = 100000  # 100 isn't enough

    # train
    for i in range(num_iterations):

        # dictionary feeding to the graph
        tf_dict = {x_tf: train_X, y_tf: train_Y}

        # Loading the previous parameters from graph
        w_prev, b_prev = sess.run([W, b])
        # Computing loss from graph (input and output data required)
        prev_loss = sess.run(loss, feed_dict=tf_dict)

        sess.run(train, feed_dict=tf_dict)
        # ^ FailedPreconditionError: Attempting to use uninitialized value beta1_power [[{{node beta1_power/read}}]]

        # check that its approaching 0 (where min occurs)
        current_loss = sess.run(loss, feed_dict=tf_dict)
        current_loss_diff = abs(current_loss - prev_loss)
        if i % 1000 == 0:
            print('\n' + ("=" * 30) + " iteration {} ".format(i) + ("=" * 30))
            print('current loss =', current_loss)
            print('current loss difference =', current_loss_diff)
            plt.title("Approximation vs training data at iteration {}".format(i))
            plt.plot(train_X, train_Y, 'ro', label='Training data')
            plt.plot(train_X, sess.run(y_pred_tf, feed_dict=tf_dict), 'g*', label='Approximation')
            # ^ learns low frequencies first, then higher
            plt.legend()
            plt.show()
        if current_loss_diff < epsilon:
            print("\nDesired accuracy reached at iteration {}!! Woop!".format(i))
            break


    plt.
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    plt.plot(train_X, sess.run(y_pred_tf, feed_dict=tf_dict), 'g*', label='Approximation')
    # ^ learns low frequencies first, then higher
    plt.legend()
    plt.show()
    sess.close()
