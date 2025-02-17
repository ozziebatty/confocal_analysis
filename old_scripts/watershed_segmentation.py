import logging
import pandas as pd
import numpy as np
from skimage import io, filters, measure, morphology, exposure
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import tifffile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting image processing...")

# Read an image from file
logging.info("Reading image...")
image = io.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/gastruloid_z.lsm')

# Normalize the image
logging.info("Normalizing image...")
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Apply Gaussian filtering to reduce noise
logging.info("Applying Gaussian filtering...")
filtered_image = gaussian(image, sigma=1, mode='constant')

# Enhance contrast using histogram equalization
logging.info("Enhancing contrast...")
enhanced_image = exposure.equalize_adapthist(filtered_image)

# Convert enhanced image to binary mask using Otsu's method
logging.info("Converting to binary mask...")
binary_mask = enhanced_image[..., 0] > filters.threshold_otsu(enhanced_image[..., 0])

# Compute distance transform
logging.info("Computing distance transform...")
distance = distance_transform_edt(binary_mask)

# Find markers as local maxima in the distance transform
logging.info("Finding markers...")
local_maxi = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=binary_mask)
markers = np.zeros_like(distance, dtype=int)
markers[tuple(local_maxi.T)] = np.arange(1, local_maxi.shape[0] + 1)

# Apply watershed segmentation
logging.info("Applying watershed segmentation...")
labels = watershed(-distance, markers, mask=binary_mask)

# Remove small objects
logging.info("Removing small objects...")
cleaned_labels = remove_small_objects(labels, min_size=100)

# Ensure cleaned_labels is of integer type for regionprops
cleaned_labels = cleaned_labels.astype(np.int32)

# Save the colored z-stack image
logging.info("Saving segmented image...")
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/watershed_segmented.tiff', cleaned_labels)

logging.info("Processing complete.")
