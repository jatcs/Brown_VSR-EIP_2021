# Jessica Turner, 7/14/2021
# Creating an agent that is trained on detailed + compressed images
# to compare learning time + loss

import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np  # will use for FFT, plotting, and saving results
import matplotlib.pyplot as plt  # to map function approximation as it learns
from PIL import Image  # Python Imaging Library
import os  # to access image files + save plots
from textwrap import wrap  # for long text on plots


# ===========================================================================
# Saving Settings - for plots, loss data - global so local functions can save
# ===========================================================================
current_directory = os.getcwd()
save_results_to = current_directory + '/Output/'
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

save_plots_to = current_directory + '/Plots/'
if not os.path.exists(save_plots_to):
    os.makedirs(save_plots_to)


# hidden nodes - provide random starting values + necessary input/output dims (shape)
def initialize_params(var_shape):
    return tf.Variable(tf.random_normal(shape=var_shape, mean=0., stddev=0.05))


# network approximation for given inputs
def Deep_NN(input_, W, B):
    # pass the inputs through the network
    A = input_  # initial activation
    n_layers = len(W)

    for i in range(n_layers - 1):
        # next activation = Weight * prev_activation + bias -> transformed to within desired range
        A = tf.sin(tf.add(tf.matmul(A, W[i]), B[i]))
    # same thing for the last layer except without restricting range
    return tf.add(tf.matmul(A, W[-1]), B[-1])


def plot_images_and_fft(images_dict, image_size=(128,128), num_rows=3, num_cols=1, fig_aspect=0.5):
    """
    Plot images and their fourier transform in the desired layout
    :param images_dict: key = image title, val = PIL Images to transform
    :param num_rows:   usually 3 - to plot the image, all fft modes and select modes for each image
    :param num_cols:   number of figures per row
    :param fig_aspect: Ratio to resize figs
    :return: None (Just plot all the figures and save the FFT values)
    """
    # image subplot script from
    # https://www.delftstack.com/howto/matplotlib/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/

    image_compare_fig, image_compare_axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                                                         figsize=plt.figaspect(fig_aspect))

    # plot image and FFT for each image
    for ind, title in enumerate(images_dict):
        image_compare_axes.ravel()[ind].imshow(images_dict[title], plt.cm.gray)
        image_compare_axes.ravel()[ind].set_title('\n'.join(wrap(title, image_size[1])), loc='center', wrap=True)
        image_compare_axes.ravel()[ind].set_axis_on()

        fft_all_modes_plot_placement = ind + num_cols
        fft_select_modes_plot_placement = ind + num_cols + num_cols

        # from function_approx_img_FFT.py
        # representing the image as a 2D function in [0,1]x[0,1]
        f_grid = np.asarray(images_dict[title], dtype=np.float32)
        f_grid = f_grid / np.max(f_grid) * 2. - 1

        # ======================================
        #   FFT 2D and display
        # ======================================
        F = np.fft.fft2(f_grid) / (image_size[0] * image_size[1] * fig_aspect)  # np.fft.fft2(f_grid) / (128 * 128 / 2)
        F = np.fft.fftshift(F)
        P_ref = np.abs(F)
        # use negative values for map space so we can see symmetry across x-y axis
        img = image_compare_axes.ravel()[fft_all_modes_plot_placement].imshow(
            P_ref, extent=[-image_size[0] // 2, image_size[0] // 2, -image_size[1] // 2, image_size[1] // 2], cmap="jet")
        image_compare_fig.colorbar(img, shrink=0.5, aspect=10,
                                   ax=image_compare_axes.ravel()[fft_all_modes_plot_placement])
        image_compare_axes.ravel()[fft_all_modes_plot_placement].set_title("Fourier modes (all)")

        # only show the first few modes
        img = image_compare_axes.ravel()[fft_select_modes_plot_placement].imshow(P_ref[54:75, 54:75],
                                                                                 extent=[-15, 15, -15, 15], cmap="jet")
        image_compare_fig.colorbar(img, shrink=0.5, aspect=10,
                                   ax=image_compare_axes.ravel()[fft_select_modes_plot_placement])
        image_compare_axes.ravel()[fft_select_modes_plot_placement].set_title("Fourier modes (main)")

        # store fft data
        np.savetxt(save_results_to + '_' + title + '_Fourier_Modes_All.csv',
                   P_ref.reshape(P_ref.shape[0], -1), fmt='%e')
        np.savetxt(save_results_to + '_' + title + '_Select_Fourier_Modes.csv',
                   P_ref[54:75, 54:75].reshape(P_ref[54:75, 54:75].shape[0], -1), fmt='%e')

    plt.tight_layout(pad=1.5)
    plt.savefig(save_plots_to + picture_name[0:exclude_extension] + '_targets.png')
    plt.show()


if __name__ == '__main__':

    # ===========================
    # Set up network architecture
    # ==========================
    num_inputs = 2  # for row and column in picture
    num_layers = 1  # number of hidden layers
    nodes_per_layer = 45
    num_outputs = 1

    # connections between layers in the neural network
    net_conn = [num_inputs] + num_layers * [nodes_per_layer] + [num_outputs]
    # set weights and biases of hidden layers
    #                       num inputs,       num_outputs
    W = [initialize_params([net_conn[i - 1], net_conn[i]]) for i in range(1, len(net_conn))]
    B = [initialize_params([net_conn[i],               1]) for i in range(len(net_conn))]

    # ================================================================================================
    # Transform original picture -> compressed version with desired quality / compression ratio
    # ================================================================================================
    picture_name = 'Stinkbug.png'  # input('Enter the name of the picture file: ')
    picture_path = os.path.join(current_directory, picture_name)
    pic_quality = 0.001  # float(input('Enter the desired picture quality to compress to (0 (worst) - 100 (best)): '))
    exclude_extension = -4  # for slicing (only including the name of the file of .png images)

    original_image = Image.open(picture_path)
    original_image.save('compressed_{}.png'.format(picture_name[0:exclude_extension]), quality=pic_quality)
    original_image = original_image.convert('L')  # 'L' for gray scale mode
    height, width = original_image.size
    newsize = (height // 5, width // 5)  # so figures can fit on single plot easier
    resized_og_image = original_image.resize(newsize)  # resize and crop
    resized_og_image = resized_og_image.crop((0, 0, np.min(newsize), np.min(newsize)))
    resized_og_image = resized_og_image.convert('L')  # for some reason it wouldn't work unless

    compressed_image = Image.open('compressed_{}.png'.format(picture_name[0:exclude_extension]))
    resized_compressed = compressed_image.resize(newsize)
    resized_compressed = resized_compressed.crop((0, 0, np.min(newsize), np.min(newsize)))
    # trying to fix fft compressed plots (forgot to convert to grey scale the first time)
    resized_compressed = resized_compressed.convert('L')

    # =========================================================================
    # see difference in detail from pre-compressed to compressed through plots
    # =========================================================================
    n_rows = 3  # 1st row = image + compressed version, 2nd row = all fourier modes for each, 3rd row = select modes
    n_cols = 2  # images per row
    resize_figaspect = 0.9  # ratio compared to original fig size

    # store the images with the formatted title
    images = {picture_name[0:exclude_extension] + ' Original Image': resized_og_image,
              picture_name[0:exclude_extension] + ' Compressed ({}% Quality)'.format(pic_quality): resized_compressed}

    plot_images_and_fft(images_dict=images, image_size=newsize, num_rows=n_rows, num_cols=n_cols,
                        fig_aspect=resize_figaspect)

    # to-do : try training on different compression ratios, compare the approximation's image and fourier transform
    # for plots
    x1_grid = np.linspace(0, 1, np.min(newsize))
    x2_grid = np.linspace(0, 1, np.min(newsize))
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    f_grid = np.asarray(resized_og_image, dtype=np.float32)   # may need to add another f_grid for resized_compressed

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

    max_iterations = 50000
    #           key = image name, value = list of the loss calculations for that image's network approximation
    loss_dict = {list(images.keys())[0]: [], list(images.keys())[1]: []}

    # train net to approx non-compressed and compressed -> compare results
    for image_name in images:
        for n in range(max_iterations):

            loss_, _ = sess.run([loss, train], feed_dict={x_train: np.hstack((X1, X2)), y_train: f})
            loss_dict[image_name].append(loss)
            if n == 1 or n % 100 == 0:
                print('Steps: %d, loss: %.3e' % (n, loss_))

            if n == 1 or n % (max_iterations // 5) == 0:
                y_pred_ = sess.run(y_pred, feed_dict={x_train: np.hstack((X1_grid.reshape(-1, 1),
                                                                          X2_grid.reshape(-1, 1)))})
                y_pred_grid = y_pred_.reshape(f_grid.shape[0], -1)

                """
                F_pred = np.fft.fft2(y_pred_grid) / (128 * 128 / 2)
                F_pred = np.fft.fftshift(F_pred)
                P_pred = np.abs(F_pred)
                
                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax = fig.add_subplot(1, 2, 1)
                # only show the first few modes
                img = plt.imshow(P_ref[54:75, 54:75], extent=[-10, 10, -10, 10], cmap="jet")
                fig.colorbar(img, shrink=0.5, aspect=10)
                plt.title("Fourier modes Reference {}".format(image_name))
                ax = fig.add_subplot(1, 2, 2)
                # only show the first few modes
                img = plt.imshow(P_pred[54:75, 54:75], extent=[-10, 10, -10, 10], cmap="jet")
                fig.colorbar(img, shrink=0.5, aspect=10)
                plt.title("Fourier modes (network)")
                
                # save plots here if needed
                # -------------------------
                plt.savefig(save_plots_to + image_name[0: -4] +
                            '_Fourier_Approx_Iteration_{}.png'.format(n))
                plt.show()
                """
                plot_images_and_fft(
                    images_dict={
                        image_name + ' Reference': resized_og_image,
                        image_name + ' Approximation (Iter {})'.format(n): y_pred_},
                    image_size=newsize,
                    num_rows=3, num_cols=1, fig_aspect=0.5
                )

                # plot the original and reconstructed images if needed
                # -------------------------
