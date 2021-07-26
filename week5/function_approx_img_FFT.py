# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:21:38 2021
@author: scai15
"""

import tensorflow as tf

if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image  # Python Imaging Library
import os


# ======================================
#   Neural network
# ======================================
# initialization for the weights / biases
def parameters_initialization(size):
    in_dim = size[0]
    out_dim = size[1]
    return tf.Variable(tf.random_normal(shape=size, mean=0., stddev=0.05))


# forward computational graph
def DNN(X, W, b):
    A = X
    L = len(W)  # number of layers
    for i in range(L - 1):
        # Matrix multiplication
        A = tf.sin(tf.add(tf.matmul(A, W[i]), b[i]))
    # the last layer - output layer
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y


# ======================================
#   Evaluation metric
# ======================================
def PSNR(original, approximated):
    max_pixel = 255.0
    original = (original + 1) / 2 * max_pixel
    approximated = (approximated + 1) / 2 * max_pixel
    mse = np.mean((original - approximated) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# ======================================
#   Main
# ======================================

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

image_name = 'sugarcane.png'  # input('What is the file name for your image?\n')
im1 = Image.open(image_name)
im1 = im1.convert('L')  # 'L' for gray scale mode
H, W = im1.size
newsize = (H // 4, W // 4)
im1 = im1.resize(newsize)  # resize and crop
im1 = im1.crop((0, 0, np.min(newsize), np.min(newsize)))
plt.imshow(im1, cmap='gray')
plt.title("Image to be approximated")
plt.show()

# representing the image as a 2D function in [0,1]x[0,1]
x1_grid = np.linspace(0, 1, np.min(newsize))
x2_grid = np.linspace(0, 1, np.min(newsize))
\P_, X2_grid = np.meshgrid(x1_grid, x2_grid)
f_grid = np.asarray(im1, dtype=np.float32)
# f_grid = f_grid / np.max(f_grid)
f_grid = f_grid / np.max(f_grid) * 2. - 1

# ======================================
#   FFT 2D and display
# ======================================
F = np.fft.fft2(f_grid) / (128 * 128 / 2)
F = np.fft.fftshift(F)
P_ref = np.abs(F)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
# use negative values for map space so we can see symmetry across x-y axis
img = plt.imshow(P_ref, extent=[-64., 64., -64., 64.], cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Fourier modes (all)")
ax = fig.add_subplot(1, 2, 2)
# only show the first few modes
img = plt.imshow(P_ref[54:75, 54:75], extent=[-10, 10, -10, 10], cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10, label='Intensity')  # label='Intensity'
plt.title("Fourier modes (main)")
# label fig with image name
plt.savefig(save_plots_to + image_name[0: -4] + '_Image_Approx_Fourier_Modes.png')
plt.show()

# Randomly select points for training
num_data = 10000
X1 = X1_grid.reshape(-1, 1)
X2 = X2_grid.reshape(-1, 1)
f = f_grid.reshape(-1, 1)
# generate the index
idx = np.random.choice(X1.shape[0], num_data, replace=False)
X1 = X1[idx, :]
X2 = X2[idx, :]
f = f[idx, :]

# ======================================
#   Building Network/Graph
# ======================================
# multi-layer neural network
num_layer = 5
num_node = 128
# the network architecture will be : 
#       [num_input] + num_layer*[num_node] + [num_output]
layers = [2] + num_layer * [num_node] + [1]
# Initialization
W = [parameters_initialization([layers[l - 1], layers[l]]) for l in range(1, len(layers))]
b = [parameters_initialization([1, layers[l]]) for l in range(1, len(layers))]

# holders for the input and output
x_train = tf.placeholder(tf.float32, shape=[None, 2])
y_train = tf.placeholder(tf.float32, shape=[None, 1])

# forward computational graph
y_pred = DNN(x_train, W, b)
# loss function
loss = tf.reduce_mean(tf.square(y_pred - y_train))

# Adam optimizer
# when you run the optimizer, the tf automatically updates the parameters
learning_rate = 6.0e-3
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Graph Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ======================================
#   Training
# ======================================
nmax = 50000
n = 0
loss_f_list = []
while n <= nmax:
    n = n + 1
    loss_, _ = sess.run([loss, train], feed_dict={x_train: np.hstack((X1, X2)), y_train: f})
    if n == 1 or n % 100 == 0:
        print('Steps: %d, loss: %.3e' % (n, loss_))

    if n == 1 or n % (nmax // 100) == 0:
        y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1_grid.reshape(-1, 1), X2_grid.reshape(-1, 1)))})
        y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)

        F_pred = np.fft.fft2(y_pred_grid) / (128 * 128 / 2)
        F_pred = np.fft.fftshift(F_pred)
        P_pred = np.abs(F_pred)

        fig, axs = plt.subplots(2, 2, figsize=plt.figaspect(0.5))
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(1, 2, 1)
        # only show the first few modes
        img = axs[0, 0].imshow(P_ref[54:75, 54:75], extent=[-10, 10, -10, 10], cmap="jet")
        fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[0, 0])
        axs[0, 0].set_title("Fourier modes (reference)", loc='center', wrap=True)
        # ax = fig.add_subplot(1, 2, 2)
        # only show the first few modes
        img = axs[0, 1].imshow(P_pred[54:75, 54:75], extent=[-10, 10, -10, 10], cmap="jet")
        fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[0, 1])
        axs[0, 1].set_title("Fourier modes (network)", loc='center', wrap=True)


        # plot the original and reconstructed images if needed
        # -------------------------
        # ax = fig.add_subplot(2, 2, 2)

        img = axs[1, 0].imshow(f_grid, cmap="gray")
        fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[1, 0])
        axs[1, 0].set_title("Target Image (Reference)", loc='center', wrap=True)

        y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1_grid.reshape(-1, 1), X2_grid.reshape(-1, 1)))})
        y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)
        img = axs[1, 1].imshow(y_pred_grid, cmap="gray")
        fig.colorbar(img, shrink=0.5, aspect=10, ax=axs[1, 1])
        axs[1, 1].set_title("Approximated image at Iteration {}".format(n), loc='center', wrap=True)

        plt.tight_layout()
        # save plots here if needed
        # -------------------------
        plt.savefig(save_plots_to + image_name[0: -4] +
                    '_Fourier_Approx_Iteration_{}.png'.format(n))
        plt.show()


# ======================================
#   Plot the final result
# ======================================
X1 = X1_grid.reshape(-1, 1)
X2 = X2_grid.reshape(-1, 1)
y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1, X2))})
y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)

# 2D plot of images
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
img = ax.imshow(f_grid, cmap="gray")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Reference image")
ax = fig.add_subplot(1, 2, 2)
img = ax.imshow(y_pred_grid, cmap="gray")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Approximated image")
plt.savefig(save_plots_to + image_name[0:-4] + '_Final_Approx.png')
plt.show()

# # 2D plot of Fourier modes
# F_pred = np.fft.fft2(y_pred_grid)/(128*128/2)   
# F_pred = np.fft.fftshift(F_pred)
# P_pred = np.abs(F_pred)
# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 2, 1)
# # only show the first few modes
# img = plt.imshow(P_ref[54:75,54:75], extent=[-10,10,-10,10], cmap="jet")
# fig.colorbar(img, shrink=0.5, aspect=10)
# plt.title("Fourier modes (reference)")
# ax = fig.add_subplot(1, 2, 2)
# # only show the first few modes
# img = plt.imshow(P_pred[54:75,54:75], extent=[-10,10,-10,10], cmap="jet")
# fig.colorbar(img, shrink=0.5, aspect=10)
# plt.title("Fourier modes (network)")
# plt.show()


# compute Peak Signal-to-Noise Ratio (PSNR)
# the higher the better
psnr = PSNR(f_grid, y_pred_grid)
print("------------------------------")
print("Final PSNR: %.3e" % (psnr))
print("------------------------------")
