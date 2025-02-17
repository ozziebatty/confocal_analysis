
import logging
import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import napari
import pandas as pd
import skimage.transform

equalise_kernel = 191
cliplimit = 0.007
scale_factor = 4

viewer = napari.Viewer()

cropping_degree = 'semi_cropped'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

#Load file_pathways
logging.info("Loading file pathways...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
raw_image = tifffile.imread(file_pathways.loc[cropping_degree, 'original'])

print(f"Image shape: {raw_image.shape} in ZCYX")

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


resized_image = resize_image(raw_image, scale_factor)
preprocessed_image = equalise(resized_image)

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer.add_image(preprocessed_image)
viewer.add_image(resized_image)


napari.run()