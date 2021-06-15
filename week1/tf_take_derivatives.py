# goal: take derivatives of a function

import tensorflow as tf
from IPython import embed
import math

x = tf.Variable(math.pi) # initial value (to check that the gradient was correct)

# use prebuilt in function (automatic differentiation)
with tf.GradientTape() as tape:
    y = tf.math.sin(x)

    # from https://www.tensorflow.org/guide/autodiff
    dy_dx = tape.gradient(y, x)

    embed() # to see if the result is correct by doing things like dy_dx.numpy()