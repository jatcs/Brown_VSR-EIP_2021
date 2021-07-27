# -*- coding: utf-8 -*-
"""
Created on 
@author: scai15
"""

import tensorflow as tf

if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
from IPython import embed

plt.rcParams['figure.figsize'] = [6, 6]

# ==========================================
# Saving Settings
# ==========================================
current_directory = os.getcwd()
save_plots_to = current_directory + '/Plots/'
if not os.path.exists(save_plots_to):
    os.makedirs(save_plots_to)

save_output_to = current_directory + '/Output/'
if not os.path.exists(save_output_to):
    os.makedirs(save_output_to)

image_name = 'Stinkbug.png'


# hidden nodes - provide random starting values + necessary input/output dims (shape)
def initialize_params(var_shape):
    return tf.Variable(tf.random_normal(shape=var_shape, mean=0., stddev=0.05))

# ===================================================
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


# network approximation for given inputs
def Deep_NN(input_, W, B):
    # pass the inputs through the network
    A = input_  # initial activation
    n_layers = len(W)

    for i in range(n_layers - 1):
        # next activation = Weight * prev_activation + bias -> transformed to within desired range
        A = tf.sin(tf.add(tf.matmul(A, W[i]), B[i]))
        # can experiment with different activations i.e. tf.nn.relu

    # same thing for the last layer except without restricting range
    return tf.add(tf.matmul(A, W[-1]), B[-1])


def compression_fft(image, ratio, is_plot=False):
    """
    :param image: original image
    :param ratio: compression ratio
    :return: compressed image
    """
    F = np.fft.fft2(image)
    F_sort = np.sort(np.abs(F.reshape(-1)))  # sort the magnitudes (small -> large)
    thresh = F_sort[int(np.floor((1 - ratio) * len(F_sort)))]  # threshold
    ind = np.abs(F) > thresh  # find indices with small magnitude
    F_compressed = F * ind  # set those as zeros
    image_compressed = np.fft.ifft2(F_compressed).real  # reconstruct an image
    if is_plot:
        plt.figure()
        plt.imshow(image_compressed, cmap='gray')
        plt.axis('off')
        plt.title('Compressed image: keep = ' + str(ratio * 100) + '%')
        plt.savefig(save_plots_to + 'Compressed_{}_keep_{}%.png'.format(image_name[0:-4], ratio * 100))
        plt.show()
    return image_compressed


original_image = imread(image_name)
original_image = np.mean(original_image, -1)
H, W = original_image.shape
newsize = (H // 2, W // 2)  # so figures can fit on single plot easier
resized_og_image = original_image[::2, ::2]  # resize and crop
resized_og_image = resized_og_image[0:np.min(newsize), 0:np.min(newsize)]

plt.figure()
plt.imshow(resized_og_image, cmap='gray')
plt.axis('off')
plt.title('Original image')
plt.show()

image_compressed_03 = compression_fft(resized_og_image, 0.2, is_plot=True)
image_compressed_01 = compression_fft(resized_og_image, 0.1, is_plot=True)
image_compressed_005 = compression_fft(resized_og_image, 0.05, is_plot=True)

# ===========================
# Set up network architecture
# ==========================
num_inputs = 2  # for row and column in picture
num_layers = 3  # number of hidden layers - add layers -> more expressive
nodes_per_layer = 45
num_outputs = 1

# connections between layers in the neural network
net_conn = [num_inputs] + num_layers * [nodes_per_layer] + [num_outputs]
# set weights and biases of hidden layers
#                       num inputs,       num_outputs
W = [xavier_init(size=[net_conn[i - 1], net_conn[i]]) for i in range(1, len(net_conn))]
B = [xavier_init(size=[1, net_conn[i]]) for i in range(1, len(net_conn))]

# for plots
x1_grid = np.linspace(0, 1, np.min(newsize))
x2_grid = np.linspace(0, 1, np.min(newsize))
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
f_grid = np.asarray(resized_og_image, dtype=np.float32)  # may need to add another f_grid for resized_compressed

# f_grid = f_grid / np.max(f_grid)
f_grid = f_grid / np.max(f_grid) * 2. - 1

x_train = tf.placeholder(tf.float32, shape=[None, 2])
y_train = tf.placeholder(tf.float32, shape=[None, 1])

# Randomly select points for training
num_points = 10000
X1 = X1_grid.reshape(-1, 1)
X2 = X2_grid.reshape(-1, 1)
f = f_grid.reshape(-1, 1)
# generate the index - why?
# idx = np.random.choice(X1.shape[0], num_points, replace=False)
# X1 = X1[idx, :]
# X2 = X2[idx, :]
# f = f[idx, :]

# forward computational graph
y_pred = Deep_NN(x_train, W, B)
# loss function
loss = tf.reduce_mean(tf.square(y_pred - y_train))

# Adam optimizer
# when you run the optimizer, the tf automatically updates the parameters
learning_rate = 6.0e-3
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Graph Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

max_iterations = 100000
#           key = image name, value = list of the loss calculations for that image's network approximation
images_dict = {'original_00': resized_og_image,
                'image_compressed_01': image_compressed_01,
               'image_compressed_03': image_compressed_03,
               'image_compressed_005': image_compressed_005}
loss_dict = {'original_00': [],
    'image_compressed_01': [], 'image_compressed_03': [], 'image_compressed_005': []}

for image_name in images_dict.keys():
    the_image = images_dict[image_name]
    F = np.fft.fft2(the_image) / (the_image.shape[0] * the_image.shape[1] / 2)
    F = np.fft.fftshift(F)
    P_ref = np.abs(F)
    f_grid = np.asarray(images_dict[image_name], dtype=np.float32)
    f = f_grid.reshape(-1, 1)
    for n in range(1, max_iterations):

        loss_, _ = sess.run([loss, train], feed_dict={x_train: np.hstack((X1_grid.reshape(-1, 1),
                                                                          X2_grid.reshape(-1, 1))),
                                                      y_train: f})

        if n == 1 or n % 100 == 0:
            loss_dict[image_name].append(loss_)
            print('Steps: %d, loss: %.3e' % (n, loss_))

        if n == 1 or n % (max_iterations // 20) == 0:
            y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1_grid.reshape(-1, 1),
                                                                      X2_grid.reshape(-1, 1)))})
            y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)

            F_pred = np.fft.fft2(y_pred_grid) / (the_image.shape[0] * the_image.shape[1] / 2)
            F_pred = np.fft.fftshift(F_pred)
            P_pred = np.abs(F_pred)

            fig, axs = plt.subplots(2, 2, figsize=plt.figaspect(0.5))
            # fig = plt.figure(figsize=plt.figaspect(0.5))
            # ax = fig.add_subplot(1, 2, 1)
            # only show the first few modes - most energetic ones
            img = axs[0, 0].imshow(P_ref[82:107, 82:107], extent=[-30, 30, -30, 30], cmap="jet")
            fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[0, 0])
            axs[0, 0].set_title("Fourier modes Reference {}".format(image_name))
            # ax = fig.add_subplot(1, 2, 2)
            # only show the first few modes
            img = axs[0, 1].imshow(P_pred[82:107, 82:107], extent=[-30, 30, -30, 30], cmap="jet")
            fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[0, 1])
            axs[0, 1].set_title("Fourier modes (network) at iter {}".format(n))

            img = axs[1, 0].imshow(images_dict[image_name], cmap="gray")
            fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[1, 0])
            axs[1, 0].set_title("Target Image (Reference) - Compressed {}%".format(float('.' + image_name[-2:]) * 100), loc='center', wrap=True)


            img = axs[1, 1].imshow(y_pred_grid, cmap="gray")
            fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[1, 1])
            axs[1, 1].set_title("Approximated image at Iteration {}".format(n), loc='center', wrap=True)

            plt.tight_layout()
            plt.savefig(save_plots_to + image_name +
                        '_Fourier_Approx_Iteration_{}.png'.format(n))
            plt.show()

    # save loss outputs
    np.savetxt(save_output_to + image_name + '_loss.txt', loss_dict[image_name], fmt='%e')
    plt.semilogy(np.arange(max_iterations / 100), loss_dict[image_name], color='green')
    plt.title('Loss over time for {}% Compressed Image Training'.format(float('.' + image_name[-2:]) * 100))
    plt.savefig(save_plots_to + image_name + 'Loss_over_time'.png)
    plt.show()

embed()
