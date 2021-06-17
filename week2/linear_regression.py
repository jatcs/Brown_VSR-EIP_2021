# Jessica Turner, 6/15/2021
import numpy as np
import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from IPython import embed
import matplotlib.pyplot as plt

range = np.random
# random batch of data from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
target_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                       7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
target_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                       2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
num_samples = target_X.shape


x_tf = tf.placeholder("float")
y_tf = tf.placeholder("float")
learning_rate = 0.45


class Node:
    def __init__(self, activation_value = 0):
        self.weight = np.random.random_sample()
        self.bias = np.random.random_sample()
        self.activation_value = activation_value
        self.previous_nodes = []
        self.next_nodes = []


class Model:
    def __init__(self, num_layers, nodes_per_layer, input_dim=1, output_dim=1):
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        pass

    def build(self):
        prev_layer = []

        for _ in range(self.num_layers):
            curr_layer = []
            for _ in range(self.nodes_per_layer):
                curr_node = Node()
                # make the layers dense
                for prev_layer_node in prev_layer:
                    curr_node.previous_nodes.append(prev_layer_node)
                    prev_layer_node.next_nodes.append(curr_node)
                # store it
                curr_layer.append(curr_node)

            self.layers.append(curr_layer)
            prev_layer = curr_layer


    def eval(self, input_val):
        # sum up all of the values (weight(activation) + bias)
        if len(self.layers) >= 0:
            for node in self.layers[0]:
                node.activation_value = input_val

        prev_result = 0
        for layer in self.layers:

            for node in layer:

                pass

            pass

        pass


    def display(self):
        # see how weights and biases for all the nodes are
        layer_num = 1
        for layer in self.layers:
            node_num = 1
            print("Layer num =", layer_num)

            for node in layer:
                print("\tNode in place {}: weight = {}, bias = {}".format((layer_num, node_num), node.weight, node.bias))
                node_num += 1

            layer_num += 1

        pass


model = Model(num_layers=1, nodes_per_layer=1, input_dim=1, output_dim=1)
model.build()
model.display()


plt.plot(target_X, target_Y, 'ro', label='Training data')
plt.legend()
plt.show()
embed()
