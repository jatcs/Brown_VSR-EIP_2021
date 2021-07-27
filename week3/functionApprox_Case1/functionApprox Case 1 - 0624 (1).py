import os

# use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import matplotlib.pyplot as plt


# ======================================
#   Function to be learned - Case 1
# ======================================
def fun_x(x):
    # five frequencies: [0.5, 1, 1.5, 2, 4]
    f = 2 * np.sin(np.pi * x) + 3 * np.sin(2 * np.pi * x) + \
        1.5 * np.sin(3 * np.pi * x) + 2.5 * np.sin(4 * np.pi * x) + 2 * np.sin(8 * np.pi * x)
    return f


# ======================================
#   Fourier transform of a function (discrete values)
# ======================================
def F_transform(fun):
    F_t = np.fft.fft(fun)
    return F_t


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
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    # the last layer - output layer
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y


# ======================================
#   Main function
# ======================================
if __name__ == "__main__":

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

    # ======================================
    #   Generating data
    # ======================================
    num_points = 201  # how many data

    # Generate training data (X,Y) -> y = f(x)
    x_in = np.linspace(-1., 1., num_points)
    x_in = np.reshape(x_in, (-1, 1))
    y_in = np.array([fun_x(i) for i in x_in])

    # The following data are used for testing and plotting
    # Generate data (X,Y) -> y = f(x), x \in (-1, 1)
    x = np.linspace(-1., 1., 129, dtype=np.float64)
    x = np.reshape(x, (-1, 1))
    y = np.array([fun_x(i) for i in x])
    # Fourier transform of the function (discrete values)
    F_y = F_transform(y[0:-1, 0])
    F_y_abs = abs(F_y)  # magnitude of a complex number
    F_y_abs = F_y_abs / (len(F_y) / 2)  # the magnitude was multiplied by the number of points
    sampling_rate = 128 / 2
    F_freq = np.fft.fftfreq(128, 1 / sampling_rate)

    # save data if needed
    np.savetxt(save_results_to + 'y_ref.txt', y, fmt='%e')
    np.savetxt(save_results_to + 'F_y.txt', F_y, fmt='%e')
    np.savetxt(save_results_to + 'F_y_abs.txt', F_y_abs, fmt='%e')

    # ======================================
    #   Building Network/Graph
    # ======================================
    # multi-layer neural network
    num_layer = 2
    num_node = 20
    # the network architecture will be : 
    #       [num_input] + num_layer*[num_node] + [num_output]
    layers = [1] + num_layer * [num_node] + [1]
    # Initialization
    W = [parameters_initialization([layers[l - 1], layers[l]]) for l in range(1, len(layers))]
    b = [parameters_initialization([1, layers[l]]) for l in range(1, len(layers))]

    # holders for the input and output
    x_train = tf.placeholder(tf.float32, shape=[None, 1])
    y_train = tf.placeholder(tf.float32, shape=[None, 1])

    # forward computational graph
    y_pred = DNN(x_train, W, b)
    # loss function
    loss = tf.reduce_mean(tf.square(y_pred - y_train))

    # Adam optimizer
    # when you run the optimizer, the tf automatically updates the parameters
    learning_rate = 5.0e-3
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
    F_y_list = []
    y_p_list = []
    plt.figure(figsize=[12, 5])
    while n <= nmax:
        n = n + 1
        loss_, _ = sess.run([loss, train], feed_dict={x_train: x_in, y_train: y_in})
        if n == 1 or n % 100 == 0:
            print('Steps: %d, loss: %.3e' % (n, loss_))

        if n == 1 or n % 1000 == 0:
            # testing and plotting
            y_pred_ = sess.run(y_pred, feed_dict={x_train: x})
            F_y_pred = F_transform(y_pred_[0:-1, 0])
            F_y_pred_abs = abs(F_y_pred) / (len(y_pred_) / 2)

            plt.subplot(1, 2, 1)
            plt.plot(x, y, 'r-')
            plt.plot(x, y_pred_, 'b.', markersize=8)
            plt.xlabel('$x$', fontsize=14)
            plt.ylabel('$y$', fontsize=14)
            plt.title('Network Approximation (Blue) vs Target Function (Red)')
            # plt.text(-0.25, -12, 'Iteration {}'.format(n))

            # plot fourier amplitude 
            # only plot the first 15 frequencies
            plt.subplot(1, 2, 2)
            plt.plot(F_freq[0:15], F_y_abs[0:15], 'ro', linewidth=2)
            plt.plot(F_freq[0:15], F_y_pred_abs[0:15], 'b*', linewidth=2)
            plt.xlabel('$freq$', fontsize=14)
            plt.ylabel('$amp$', fontsize=14)
            plt.title('Fourier Domain - Frequencies + Intensities Learned')

            plt.savefig(save_plots_to + 'result_step_%d.png' % (n))
            plt.savefig(save_plots_to + 'result_step_%d.eps' % (n))
            plt.clf()

            # check convergence

        loss_f_list.append(loss_)

    # ======================================
    #   Plot the final result
    # ======================================
    y_pred_ = sess.run(y_pred, feed_dict={x_train: x})
    F_y_pred = F_transform(y_pred_[0:-1, 0])
    F_y_pred_abs = abs(F_y_pred) / (len(y_pred_) / 2)

    np.savetxt(save_results_to + 'y_pred.txt', y_pred_, fmt='%e')
    np.savetxt(save_results_to + 'F_y_pred.txt', F_y_pred, fmt='%e')
    np.savetxt(save_results_to + 'F_y_pred_abs.txt', F_y_pred_abs, fmt='%e')

    plt.figure(figsize=[12, 5])
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'r-')
    plt.plot(x, y_pred_, 'b.', markersize=8)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('Network Approximation Final Result')

    plt.subplot(1, 2, 2)
    # plotting the actual frequencies and their magnitude #
    plt.plot(F_freq[0:15], F_y_abs[0:15], 'ro', linewidth=2)
    # plotting the predicted frequencies + magnitudes
    plt.plot(F_freq[0:15], F_y_pred_abs[0:15], 'b*', linewidth=2)
    plt.xlabel('$freq$', fontsize=14)
    plt.ylabel('$amp$', fontsize=14)
    plt.title('Predicted Frequencies and Magnitudes')
    plt.savefig(save_plots_to + 'result_final.png')
    plt.show()

    # ======================================
    #   Plot the expanded signals and the reconstructed signal
    # ======================================
    # why only plot these particular indeces for frequencies?
    y1 = np.array([F_y_pred_abs[1] * np.sin(2 * np.pi * F_freq[1] * i) for i in x])
    y2 = np.array([F_y_pred_abs[2] * np.sin(2 * np.pi * F_freq[2] * i) for i in x])
    y3 = np.array([F_y_pred_abs[3] * np.sin(2 * np.pi * F_freq[3] * i) for i in x])
    y4 = np.array([F_y_pred_abs[4] * np.sin(2 * np.pi * F_freq[4] * i) for i in x])
    y8 = np.array([F_y_pred_abs[8] * np.sin(2 * np.pi * F_freq[8] * i) for i in x])

    y_recons = y1 + y2 + y3 + y4 + y8

    plt.figure(figsize=[12, 12])
    plt.subplot(3, 2, 1)
    plt.plot(x, y_recons, 'r-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('Reconstructed signal', fontsize=14)

    plt.subplot(3, 2, 2)
    plt.plot(x, y1, 'b-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('y1', fontsize=14)
    plt.subplot(3, 2, 3)
    plt.plot(x, y2, 'b-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('y2', fontsize=14)
    plt.subplot(3, 2, 4)
    plt.plot(x, y3, 'b-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('y3', fontsize=14)
    plt.subplot(3, 2, 5)
    plt.plot(x, y4, 'b-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('y4', fontsize=14)
    plt.subplot(3, 2, 6)
    plt.plot(x, y8, 'b-')
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$y$', fontsize=14)
    plt.title('y8', fontsize=14)

    plt.savefig(save_plots_to + 'result_final_signals.png')
    plt.show()
    plt.clf()


    loss_f_list = np.asarray(loss_f_list).reshape(-1, 1)
    # ValueError: x and y must have same first dimension, but have shapes (1,) and (50001,)
    loss_x = np.arange(nmax + 1)
    plt.semilogy(loss_x[:6000], loss_f_list[:6000], 'orange', linewidth=2)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.title('Loss over training time')
    plt.savefig(save_plots_to + 'loss_plot.png')
    plt.show()