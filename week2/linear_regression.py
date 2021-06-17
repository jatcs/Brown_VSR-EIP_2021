# Jessica Turner, 6/15/2021-6/17/2021
import numpy as np

import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from IPython import embed
import matplotlib.pyplot as plt

# range = np.random
# random batch of data from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb
target_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                       7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
target_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                       2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
num_samples = target_X.shape

'''
x_tf = tf.placeholder("float")
y_tf = tf.placeholder("float")
learning_rate = 0.45
'''


class Node:
    def __init__(self, activation_value=0):
        self.weight = tf.Variable(np.random.random_sample())
        self.bias = tf.Variable(np.random.random_sample())
        self.activation_value = activation_value
        self.previous_nodes = []
        self.next_nodes = []

        # allow access of variable values
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


class Model:
    def __init__(self, num_layers, nodes_per_layer, input_dim=1, output_dim=1):
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []
        pass

    def build(self):
        """
        Creates a densely connected neural network with the specified dimensions
        :return: None (self.layers built)
        """

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

            # move on to the next set
            self.layers.append(curr_layer)
            prev_layer = curr_layer

    def eval(self, input_val):
        """
        Evaluates the current line (corresponding ouptuts) based on the weights and biases
        :param input_val:
        :return:
        """
        if not isinstance(input_val, list):
            input_val = np.asarray(input_val)

        outputs = []
        for val in input_val:
            # sum up all of the values (weight(activation) + bias)
            if len(self.layers) >= 0:
                for node in self.layers[0]:
                    node.activation_value = val

            total_eval = 0
            for layer in self.layers:
                layer_sum = 0
                for node in layer:
                    # node.weight * node.activation_value + node.bias
                    node_eval = \
                    (node.sess.run(node.weight) * node.activation_value \
                     + node.sess.run(node.bias))
                    layer_sum += node_eval
                    for next_node in node.next_nodes:
                        next_node.activation_value += node_eval
                total_eval += layer_sum
                pass

            outputs.append(total_eval)

        return np.asarray(outputs)

    def display(self):
        # see how weights and biases for all the nodes are
        layer_num = 1
        for layer in self.layers:
            node_num = 1
            print("Layer num =", layer_num)

            for node in layer:
                print(
                    "\tNode in place {}: weight = {}, bias = {}".format(
                        (layer_num, node_num),
                        node.sess.run(node.weight),
                        node.sess.run(node.bias)))
                node_num += 1

            layer_num += 1

        print('Current Loss =', self.loss())
        pass

    def loss(self):
        total_loss = 0

        for index in range(num_samples[0]):
            for y_guess in self.eval(target_X):
                total_loss += tf.square(y_guess - target_Y[index])
        return total_loss / num_samples

        # alternative so its a tensor which can calculate gradient hopefully
        # return tf.reduce_mean(tf.square(self.eval(target_X) - target_Y))

    def train(self):
        """
        Update node weights and biases based on training data and loss
        :return: None
        """
        for layer in self.layers:
            for node in layer:
                loss = self.loss()
                dJ_dw = tf.gradients(loss, node.weight)[0]
                dJ_db = tf.gradients(loss, node.bias)[0]
                print("dJ_dw =", dJ_dw)
                print("dJ_db =", dJ_db)
                # outputs "None" :(
        pass


model = Model(num_layers=1, nodes_per_layer=3, input_dim=1, output_dim=1)
model.build()
model.display()
model.train()


plt.plot(target_X, target_Y, 'ro', label='Training data')

plt.plot(target_X, model.eval(target_X), 'green', label='Approximation')
plt.legend()
plt.show()
# embed()
