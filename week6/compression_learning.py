# Jessica Turner, 7/14/2021
# Creating an agent that is trained on detailed + compressed images
# to compare learning time

import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np                          # will use for FFT
import matplotlib.pyplot as plt             # to map function approximation as it learns
from PIL import Image                       # Python Imaging Library
import os                                   # to access image files + save plots


# hidden nodes - provide random starting values + necessary input/output dims (shape)
def initialize_params(var_shape):
    return tf.Variable(tf.random_normal(shape=var_shape, mean=0., stddev=0.05))


# network approximation for given inputs
def Deep_NN(input_, W, B):
    # pass the inputs through the network
    A = input_  # initial activation
    n_layers = len(W)

    for i in range(n_layers - 1):
        # next activation = Weight * prev_activation + bias -> transformed to within desired range
        A = tf.sin(tf.add(tf.matmul(A, W[i]), B[i]))
    # same thing for the last layer except without restricting range
    return tf.add(tf.matmul(A, W[-1]), B[-1])


if __name__ == '__main__':

    # ===========================
    # Set up network architecture
    # ==========================
    num_inputs = 2  # for row and column in picture
    num_layers = 1  # number of hidden layers
    nodes_per_layer = 45
    num_outputs = 1

    # connections between layers in the neural network
    net_conn = [num_inputs] + num_layers * [nodes_per_layer] + [num_outputs]
    # set weights and biases of hidden layers
    #                      num inputs,      num_outputs
    W = [initialize_params([net_conn[i - 1], net_conn[i]]) for i in range(1, len(net_conn))]
    B = [initialize_params([net_conn[i],               1]) for i in range(len(net_conn))]

    # ================================================================================================
    # Get input data - original picture -> compressed version with desired quality / compression ratio
    # ================================================================================================
    picture_name = input('Enter the name of the picture file: ')

    pass
