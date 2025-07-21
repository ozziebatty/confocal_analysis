import logging
import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import napari
import skimage.transform
import pandas as pd
import os
from skimage import io


input_folder = '/Users/oskar/Desktop/exp024gomb_SBSO_OPP_d5/raw_images'
output_folder = '/Users/oskar/Desktop/exp024gomb_SBSO_OPP_d5/preprocessed_images'

cliplimit = 0.007
equalise_kernel = 191
scale_factor = 4
#Use as low as possible without introducing artefacts from brightness, typically 20 or higher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

def resize_image(image, scale_factor):
    logging.info("Resizing...")

    # Calculate new shape
    y_pixels = image.shape[2]
    x_pixels = image.shape[3]
    total_z = image.shape[0]
    total_channels = image.shape[1]
    
    new_shape = (total_z, total_channels, y_pixels * scale_factor, x_pixels * scale_factor)
    
    # Resize image
    resized_image = np.zeros(new_shape, dtype=np.float32)
    
    for z in range(total_z):
        z_slice = image[z]
        for channel in range(total_channels):
            channel_slice = z_slice[channel]
            channel_slice = skimage.transform.resize(channel_slice, (y_pixels * scale_factor, x_pixels * scale_factor), anti_aliasing=True)
            resized_image[z][channel] = channel_slice

    resized_normalized = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())
    image_uint8 = (resized_normalized * 255).astype(np.uint8)
    resized_image = image_uint8

    return resized_image

def equalise(image):
    logging.info("Equalising...")

    y_pixels = image.shape[2]
    x_pixels = image.shape[3]
    total_z = image.shape[0]
    total_channels = image.shape[1]

    equalised_stack = np.zeros_like(image, dtype=np.uint8)
    
    for z in range(total_z):
        z_slice = image[z]

        for channel in range(total_channels):
            channel_slice = z_slice[channel]
            channel_slice_normalized = (channel_slice - channel_slice.min()) / (channel_slice.max() - channel_slice.min())

            equalised = img_as_ubyte(equalize_adapthist(channel_slice_normalized, kernel_size=(equalise_kernel,equalise_kernel), clip_limit=cliplimit, nbins=256))
        
            z_slice[channel] = equalised

        equalised_stack[z] = z_slice

    return equalised_stack


def process_folder(input_folder, output_folder, scale_factor):
    # Create the output directory, including any intermediate directories
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Creating output folder...")

    # Iterate over each file in the input directory
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if it's an image file (adjust as needed for your format, e.g., '.tif', '.png')
        if filename.endswith(('.tif', '.png', '.jpg', '.tiff')) and os.path.isfile(input_path):
            print(f"Processing {filename}...")

            # Load the image
            image = tifffile.imread(input_path)

            #Resize the image
            resized_image = resize_image(image, scale_factor)

            # Apply the equalise function to the image
            preprocessed_image = equalise(resized_image)
            
            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, filename)
            tifffile.imwrite(output_path, preprocessed_image)
            print(f"Saved processed image to {output_path}")


process_folder(input_folder, output_folder, scale_factor)

logging.info("Complete")

