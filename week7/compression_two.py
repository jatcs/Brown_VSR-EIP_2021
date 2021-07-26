from PIL import Image  # attempt to use Image.save(quality=compression_percentage)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
if tf.__version__ > "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import os               # save plots and get image path
from IPython import embed

def sort_high_low(a_list):
    """
    Helper function to remove high frequencies from 2D FFT (diy compression)
    can add conditions to change shape from [[]] to plain list if necessary - then change it back to
    :param a_list: The list to sort
    :return: Sorted list from highest to lowest
    """
    if len(a_list) == 0:
        return []
    elif len(a_list) == 1:
        return a_list

    # put all the high numbers in the front by just
    # taking out the max, putting it at the front and
    # rerunning on max-removed list
    curr_max = max(a_list)
    max_index = a_list.index(curr_max)
    ret_list = a_list[0: max_index]
    # bounds check (if the max index is at the end of the list or not (add extra terms as needed))
    if max_index != len(a_list) - 1:
        ret_list += a_list[max_index + 1: len(a_list)]

    return [curr_max] + sort_high_low(ret_list)


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


# ==========================================
# Get and compress target image
# ==========================================
compression_percentage = 1
image_name = 'sugarcane.png'
exclude_extension = -4
image_path = os.path.join(current_directory, image_name)
target_image = Image.open(image_path)
# compress + save compressed version in another file
target_image.save('compressed_{}.png'.format(image_name[0: exclude_extension]), quality=compression_percentage)
compressed_image = Image.open('compressed_{}.png'.format(image_name[0:exclude_extension]))

# ==========================================================
# Plot both images and their fourier modes to see differences
# ===========================================================
fig_aspect = 0.5
fig, axs = plt.subplots(2, 1, figsize=plt.figaspect(fig_aspect))
axs[0].imshow(target_image)
axs[0].set_title('Starting image', loc='center')
axs[1].imshow(compressed_image)
axs[1].set_title('Compressed Image', loc='center')

plt.tight_layout()

plt.savefig(save_plots_to + image_name[0: exclude_extension] + '_target_compressed.png')
plt.show()

# try manually compressing by removing high frequencies
f_grid = np.asarray(target_image, dtype=np.float32)
image_size = f_grid.shape
f_grid = f_grid / np.max(f_grid) * 2. - 1

# ======================================
#   FFT 2D and display
# ======================================
F = np.fft.fft2(f_grid) / (image_size[0] * image_size[1] * fig_aspect)  # np.fft.fft2(f_grid) / (128 * 128 / 2)
F = np.fft.fftshift(F)
P_ref = np.abs(F)
og_shape = P_ref.shape
temp = list(P_ref.reshape(-1,))
temp.sort(reverse=True)

# take out highest few frequencies
amount_to_remove = len(temp) // 4
temp_max = max(temp[amount_to_remove:])
temp = amount_to_remove * [temp_max] + temp[amount_to_remove:]
temp.sort()
P_ref = np.asarray(temp)
# P_ref = P_ref.reshape(len(temp) - amount_to_remove, og_shape[1], og_shape[2])
P_ref = P_ref.reshape(og_shape)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
# use negative values for map space so we can see symmetry across x-y axis
img = plt.imshow(P_ref, extent=[-10, 10, -10, 10],
                 cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10, label='Intensity')
plt.title("Compressed FFT")

ax = fig.add_subplot(1, 2, 2)
# only show the first few modes
iF = np.fft.ifft2(P_ref).real
img = plt.imshow(iF, extent=[-image_size[0] // 2, image_size[0] // 2, -image_size[1] // 2, image_size[1] // 2],
                 cmap="jet")
fig.colorbar(img, shrink=0.5, aspect=10)
plt.title("Reconstructed Image")
# label fig with image name
plt.savefig(save_plots_to + image_name[0: exclude_extension] + '_Compressed_Fourier_Modes.png')
plt.show()


embed()