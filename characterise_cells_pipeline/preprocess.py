from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
import sys
from skimage import img_as_ubyte
import os
from skimage.filters import gaussian

dapi_clip_limit = 0.08
fluorescence_clip_limit = 0.02
gaussian_sigma = 0.5
gaussian_kernel_size = 31

clahe_kernel_size = 31

n_bins = 256



debug_mode = True
remove_channel = 'False'

if debug_mode == True:
    input_folder_pathway = '/Users/oskar/Desktop/format_test/'
    output_folder_pathway = '/Users/oskar/Desktop/format_test/'
    dapi_channel = 0 #Post removal
    input_file_name = 'SBSO_stellaris.tiff'
    output_file_name = 'SBSO_stellaris_preprocessed.tiff'
    remove_channel = 'False'
    channel_to_remove = 2
    display_napari = 'True'
else:
    input_path = sys.argv[1]
    output_folder_pathway = sys.argv[2]
    cpu_num = ''#sys.argv[3]
    dapi_channel = int(sys.argv[4])
    display_napari = sys.argv[5]


channels_to_preprocess = [0, 1, 2, 3]
#output_path = output_folder_pathway + '/preprocessed' + cpu_num + '.tiff'
output_path =  output_folder_pathway + '/' + output_file_name
input_path = input_folder_pathway + '/' + input_file_name
image = tifffile.imread(input_path)

print("Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

def EAH(channel_slice, channel):
    if channel == dapi_channel:
        clip_limit = dapi_clip_limit
    else:
        clip_limit = fluorescence_clip_limit
        
    equalised_stack = img_as_ubyte(equalize_adapthist(
        channel_slice, 
        kernel_size=clahe_kernel_size, 
        clip_limit=clip_limit, 
        nbins=n_bins
    ))

    #equalised_stack = rearranged_equalised_stack.transpose()

    return equalised_stack


def apply_gaussian(channel_slice, gaussian_sigma, gaussian_kernel_size):
    """Apply Gaussian blur and CLAHE preprocessing to image"""

    gaussian_blurred_image = img_as_ubyte(gaussian(channel_slice, sigma=gaussian_sigma, truncate=gaussian_kernel_size))
    
    return gaussian_blurred_image


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
    print("Processing channel", channel)
    channel_slice = image[:, channel, :, :]
    rearranged_channel_slice = channel_slice.transpose()
    equalised_channel_slice = EAH(channel_slice, channel)
    if channel == dapi_channel:
        gaussian_blurred_channel_slice = apply_gaussian(equalised_channel_slice, gaussian_sigma, gaussian_kernel_size)
        preprocessed_channel_slice = gaussian_blurred_channel_slice
    else:
        preprocessed_channel_slice = equalised_channel_slice
    preprocessed_image[:, channel, :, :] = preprocessed_channel_slice

save(preprocessed_image, output_path)

display(image, preprocessed_image)

print("Preprocessing succesfully complete")