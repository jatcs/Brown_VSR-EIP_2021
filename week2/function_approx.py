# week 2 assignment, 6/16/2021
# Jessica Turner
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def function_target(x):
    return (5 + tf.sin(x) + tf.sin(2 * x) + tf.sin(3 * x) + tf.sin(4 * x)) \
        if (x < 0) else tf.cos(10 * x)


# constants
X_DIM, LOWER_BOUND = 0, 0
Y_DIM, UPPER_BOUND = 1, 1

# initialize training set
num_points = 50
domain = (-1 * math.pi, math.pi)
train_X = ((domain[UPPER_BOUND] - domain[LOWER_BOUND]) * np.random.random_sample(num_points)) + domain[LOWER_BOUND]
train_Y = np.asarray([function_target(x) for x in train_X])


class Agent:
    def __init__(self):
        self.target_function = function_target
        self.learning_rate = 0.01
        self.desired_accuracy = 0.0001
        self.num_iterations = 100


if __name__ == '__main__':
    print("Testing function target/training data")
    plt.plot(train_X, train_Y, 'ro', label='Training data')
    plt.legend()
    plt.show()
