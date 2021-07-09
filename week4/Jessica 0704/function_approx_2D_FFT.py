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
import matplotlib.pyplot as plt
import os # for saving plots


# ======================================
#   Function to be learned
# ======================================
def function_2D(x1, x2):
    # f = np.sin(2*np.pi*x1) + np.sin(2*np.pi*x2) + np.sin(2*np.pi*(x1+x2) )
    f = np.sin(2*np.pi*x1) + np.sin(3*np.pi*x1) + 2*np.sin(2*np.pi*x2) + 0.5*np.sin(2*np.pi*(x1+x2) )
    return f

def plot_2Dfun(x, y, z, xs=None, ys=None, zs=None):    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if xs is not None:
        ax.scatter(xs, ys, zs, marker='o', color = "k", s=30)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap="jet", linewidth=0)
    # Customize the z axis.
    ax.zaxis.set_major_formatter('{x:.01f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.title("Function to be approximated")
    ax.set_xlabel("X1 axis")
    ax.set_ylabel("X2 axis")
    ax.set_zlabel("Y axis")
    if not os.path.exists('2D_function_plot.png'):
        plt.savefig('2D_function_plot.png')
    plt.show()


# ======================================
#   Neural network
# ======================================
# initialization for the weights / biases
def parameters_initialization(size):
    in_dim = size[0]
    out_dim = size[1]
    return tf.Variable(tf.random_normal(shape=size, mean = 0., stddev = 0.05))
# forward computational graph
def DNN(X, W, b):
    A = X
    L = len(W)  # number of layers
    for i in range(L - 1):
        # Matrix multiplication
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    # the last layer - output layer
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y



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

# Data on a regular grid
x1_grid = np.linspace(0, 1, 128)
x2_grid = np.linspace(0, 1, 128)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid) # join together axes
f_grid = function_2D(X1_grid, X2_grid)
# plot_2Dfun(X1_grid, X2_grid, f_grid)

# ======================================
#   FFT 2D and display
# ======================================
F = np.fft.fft2(f_grid)/(128*128/2)   
F = np.fft.fftshift(F)
P_ref = np.abs(F)   # is this because the negative amplitudes are basically unimportant?
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
img = plt.imshow(P_ref, extent = [-64.,64.,-64.,64.] , cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Fourier modes (all)")
ax = fig.add_subplot(1, 2, 2)
# only show the first few modes
img = plt.imshow(P_ref[59:70,59:70], extent=[-5,5,-5,5] , cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Fourier modes (main)")

if not os.path.exists(save_plots_to + 'Approx_before_training.png'):
    plt.savefig(save_plots_to + 'Approx_before_training.png')
plt.show()


# Randomly select points for training
num_data = 100
X1 = X1_grid.reshape(-1,1)
X2 = X2_grid.reshape(-1,1)
f = f_grid.reshape(-1,1)
# generate the index
idx = np.random.choice(X1.shape[0], num_data, replace=False)
X1 = X1[idx, :]
X2 = X2[idx, :]
f = f[idx, :]

# plot the data
plot_2Dfun(X1_grid, X2_grid, f_grid, X1, X2, f)


# ======================================
#   Building Network/Graph
# ======================================
# multi-layer neural network
num_layer = 2
num_node = 20
# the network architecture will be : 
#       [num_input] + num_layer*[num_node] + [num_output]
layers = [2] + num_layer*[num_node] + [1]
# Initialization
W = [parameters_initialization([layers[l-1], layers[l]]) for l in range(1, len(layers))]
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
learning_rate = 1.0e-3
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Graph Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())




# ======================================
#   Training
# ======================================
nmax = 5000
n = 0
loss_f_list = []
while n <= nmax:
    n = n + 1
    loss_, _ = sess.run([loss, train], feed_dict={x_train: np.hstack((X1,X2)), y_train: f})
    if n==1 or n%100 == 0:
        print('Steps: %d, loss: %.3e'%(n, loss_))
        
    if n==1 or n%1000 == 0:
        y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1_grid.reshape(-1,1),X2_grid.reshape(-1,1)))})
        y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)
        
        F_pred = np.fft.fft2(y_pred_grid)/(128*128/2)   
        F_pred = np.fft.fftshift(F_pred)
        P_pred = np.abs(F_pred)
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1)
        # only show the first few modes
        plt.title('Actual Frequencies')
        ax.set_xlabel('X1 axis')
        ax.set_ylabel('X2 axis')
        img = plt.imshow(P_ref[59:70,59:70], extent=[-5,5,-5,5] , cmap="jet")
        fig.colorbar(img, shrink=0.5, aspect=10)
        # caption explaining why we only display a few modes
        fig.text(-4, -4, "Only showing a few modes of this transform to show the extreme frequencies", wrap=True)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_xlabel('X1 axis')
        ax.set_ylabel('X2 axis')
        plt.title("Approximation Frequencies at iteration {}".format(n),
                  loc='left', wrap=True)
        # only show the first few modes
        img = plt.imshow(P_pred[59:70,59:70], extent=[-5,5,-5,5] , cmap="jet")
        fig.colorbar(img, shrink=0.5, aspect=10)
        
        # save plots here if needed
        # -------------------------
        plt.savefig(save_plots_to + '2D_Fourier_Transform_result_step_%d.png' % n)


        plt.show()

# ======================================
#   Plot the final result
# ======================================

X1 = X1_grid.reshape(-1,1)
X2 = X2_grid.reshape(-1,1)
y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1,X2))})
y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)


# 2D plot

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
plt.title('Reference (actual 2D plot)')
img = ax.imshow(f_grid, cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
ax = fig.add_subplot(1, 2, 2)
plt.title('Network Approximation after training')
img = ax.imshow(y_pred_grid, cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.show() # is this just the 2d version of the 3d graph?


# 2D plot
F_pred = np.fft.fft2(y_pred_grid)/(128*128/2)   
F_pred = np.fft.fftshift(F_pred)
P_pred = np.abs(F_pred)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel('X1 axis')
ax.set_ylabel('X2 axis')
plt.title('Reference (actual 2D plot)')
# only show the first few modes
img = plt.imshow(P_ref[59:70,59:70], extent=[-5,5,-5,5] , cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)

ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel('X1 axis')
ax.set_ylabel('X2 axis')
plt.title('Network Approximation after training')
# only show the first few modes
img = plt.imshow(P_pred[59:70,59:70], extent=[-5,5,-5,5] , cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.savefig(save_plots_to + 'Final_Result_Actual_vs_Approx.png')
plt.show()