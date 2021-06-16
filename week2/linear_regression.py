# Jessica Turner, 6/15/2021
import numpy as np
import tensorflow as tf
from IPython import embed
import matplotlib.pyl

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

# choose random starting values for the parameters


embed()
