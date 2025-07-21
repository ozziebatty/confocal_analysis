from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
import sys
from skimage import img_as_ubyte
import os

equalise_kernel = (16,16,10)
clip_limit = 0.005
n_bins = 256

debug_mode = False
remove_channel = 'False'

if debug_mode == True:
    input_folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/'
    output_folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/'
    dapi_channel = 2 #Post removal
    input_file_name = 'SBSE_example_image_semi_cropped.tiff'
    output_file_name = 'SBSE_example_image_top_preprocessed.tiff'
    remove_channel = 'True'
    channel_to_remove = 2
    display_napari = 'True'
else:
    input_path = sys.argv[1]
    output_folder_pathway = sys.argv[2]
    cpu_num = ''#sys.argv[3]
    dapi_channel = int(sys.argv[4])
    display_napari = sys.argv[5]


#channels_to_preprocess = [0, 1, 2, 3]
output_path = output_folder_pathway + '/preprocessed' + cpu_num + '.tiff'
image = tifffile.imread(input_path)

print("Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

def EAH(channel_slice):
    rearranged_equalised_stack = img_as_ubyte(equalize_adapthist(rearranged_channel_slice, kernel_size=(equalise_kernel), clip_limit=clip_limit, nbins=n_bins))
    equalised_stack = rearranged_equalised_stack.transpose()

    return equalised_stack

def save(image, output_path):
    tifffile.imwrite(output_path, image)


def display(image, preprocessed_image):
    if display_napari == 'True':
        import napari
        viewer = napari.Viewer()
        viewer.add_image(image)
        viewer.add_image(preprocessed_image)
        napari.run()


if remove_channel == 'True':
    image = image[:, np.delete(np.arange(image.shape[1]), channel_to_remove), :, :]

total_z = image.shape[0]
total_channels = image.shape[1]

preprocessed_image = image.copy()   

for channel in range(total_channels):
    channel_slice = image[:, channel, :, :]
    rearranged_channel_slice = channel_slice.transpose()
    preprocessed_channel_slice = EAH(channel_slice)
    preprocessed_image[:, channel, :, :] = preprocessed_channel_slice

save(preprocessed_image, output_path)

display(image, preprocessed_image)

print("Preprocessing succesfully complete")