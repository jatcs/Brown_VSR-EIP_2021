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

    ret_list = []
    # put all the high numbers in the front
    curr_max = max(a_list)
    max_index = a_list.index(curr_max)
    if max_index == len(a_list) - 1:
        ret_list = a_list[0: max_index]
    else:
        ret_list = a_list[0: max_index] + a_list[max_index + 1: len(a_list)]
    return [curr_max] + sort_high_low(ret_list)


"""
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
fig, axs = plt.subplots(2, 1, figsize=plt.figaspect(0.5))
axs[0].imshow(target_image)
axs[0].set_title('Starting image', loc='center')
axs[1].imshow(compressed_image)
axs[1].set_title('Compressed Image', loc='center')

plt.tight_layout()

plt.savefig(save_plots_to + image_name[0: exclude_extension] + '_target_compressed.png')
plt.show()
"""
ex = list(np.random.randint(5, size=10))
print(ex)
sorted_ex = sort_high_low(ex)
print(sorted_ex)

embed()