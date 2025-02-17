import logging
import numpy as np
import tifffile
from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import skimage.transform
from scipy.ndimage import zoom
import cv2
import napari
import csv
import pandas as pd

cropping_degree = 'semi_cropped'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting image processing...")

#Load image
logging.info("Loading file pathways...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
image = tifffile.imread(file_pathways.loc[cropping_degree, 'original_tiff'])
print(f"Image shape: {image.shape} in ZCYX")

#Rescale
logging.info("Rescaling...")
scale_factor = 1
zoom_factor_z = scale_factor*0.9/1.25
resampled_image = zoom(image, (zoom_factor_z, 1, 1, 1), order=3)  # Cubic interpolation

image = resampled_image

# Obtain image properties
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

channel_to_segment = 0


def preprocess(image):
    logging.info("Preprocessing...")
    preprocessed_image = np.zeros((total_z, y_pixels, x_pixels), dtype=np.int32)

    resized_image = np.zeros((total_z, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.float32)
    
    for z in range(total_z):
        z_slice = image[z][channel_to_segment]

        resized_z_slice = skimage.transform.resize(z_slice, (y_pixels * scale_factor, x_pixels * scale_factor), anti_aliasing=True)

        z_slice_in_process = resized_z_slice

        equalized = img_as_ubyte(equalize_adapthist(z_slice_in_process, kernel_size=None, clip_limit=0.01, nbins=256))
        equalized_twice = img_as_ubyte(equalize_adapthist(equalized, kernel_size=None, clip_limit=0.01, nbins=256))
        equalized_thrice = img_as_ubyte(equalize_adapthist(equalized_twice, kernel_size=None, clip_limit=0.01, nbins=256))

        resized_image[z] = equalized_thrice

    return resized_image

def segment_3D(image):
    # Initialize Cellpose
    logging.info("Initializing Cellpose...")
    model = models.Cellpose(gpu=False, model_type='cyto2')
    
    # Perform 3D segmentation on the entire stack
    logging.info("Segmenting 3D stack...")
    segmented, flows, styles, diams = model.eval(
        image,
        diameter= (8.6 * scale_factor),  # Adjust as needed
        flow_threshold=0.6,  # Adjust as needed
        cellprob_threshold=0.4,  # Adjust as needed
        do_3D=True  # Set this flag to perform 3D segmentation
    )

    return segmented

preprocessed_image = preprocess(image)
print(f"Image shape: {preprocessed_image.shape} in ZCYX")
segmented = segment_3D(preprocessed_image)
print("Total cells segmented: ", len(np.unique(segmented)))

# Save the labeled masks to a TIFF file
tifffile.imwrite((file_pathways.loc[cropping_degree, 'segmented_tiff']), segmented.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_labels(segmented)
viewer.add_image(preprocessed_image)
napari.run()
