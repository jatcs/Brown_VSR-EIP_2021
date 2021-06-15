import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# input variable
x = tf.Variable(1.0) # 1 is the starting value

# nodes
a = 2 * x + 3
b = 6 * a
y = b + 2

dy_db = tf.gradients(y, b)[0]
dy_da = tf.gradients(y, a)[0]
dy_dx = tf.gradients(y, x)[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("==================================\n" +
          ("a = {}" + "\n" +
           "b = {}" + "\n" +
           "y = {}").format(
              sess.run(a), sess.run(b), sess.run(y)) + "\n"
          "==================================")