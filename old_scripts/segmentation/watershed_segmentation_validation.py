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
from matplotlib.animation import FuncAnimation
import tifffile
from skimage.segmentation import find_boundaries
from matplotlib.widgets import Slider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting image processing...")

# Read an image from file
logging.info("Reading image...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/very_cropped_gastruloid_z.tiff')

# Normalize the image
logging.info("Normalizing image...")
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Apply Gaussian filtering to reduce noise
logging.info("Applying Gaussian filtering...")
filtered_image = gaussian(image, sigma=1, mode='constant')

# Enhance contrast using histogram equalization
logging.info("Enhancing contrast...")
enhanced_image = exposure.equalize_adapthist(filtered_image)

# Convert enhanced image to binary mask using Otsu's method across the entire stack
logging.info("Converting to binary mask...")
binary_mask = enhanced_image > filters.threshold_otsu(enhanced_image)

# Ensure that the binary_mask is now 3D
logging.info(f"binary_mask shape after thresholding: {binary_mask.shape}")
# Compute distance transform
logging.info("Computing distance transform...")
distance = distance_transform_edt(binary_mask)

# Verify that the binary_mask and distance are 3D
logging.info(f"binary_mask shape: {binary_mask.shape}")
logging.info(f"distance shape: {distance.shape}")

assert binary_mask.ndim == 3, "binary_mask should be 3D"
assert distance.ndim == 3, "distance array should be 3D"

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

# Generate a random color map
np.random.seed(42)  # For reproducibility
unique_labels = np.unique(cleaned_labels)
color_map = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)

# Create an RGB image where each label is replaced by its corresponding color
colored_labels = np.zeros((*cleaned_labels.shape, 3), dtype=np.uint8)

for i, label in enumerate(unique_labels):
    colored_labels[cleaned_labels == label] = color_map[i]

# Save the colored z-stack image
logging.info("Saving segmented image...")
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/watershed_segmented_very_cropped.tiff', colored_labels)

logging.info("Processing complete.")

# Generate boundaries (outlines) of the segmented labels
boundaries = find_boundaries(cleaned_labels, mode='thick')

# Create random colors for each segmented region, skipping label 0
np.random.seed(42)  # For reproducibility
unique_labels = np.unique(cleaned_labels)
color_map = {label: np.random.randint(0, 255, size=3) for label in unique_labels if label != 0}

# Initial display setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.imshow(image[0], cmap='gray')

# Slider setup
ax_slice = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slice, 'Slice', 0, image.shape[0]-1, valinit=0, valstep=1)

# Update function for the slider
def update(val):
    slice_index = int(slider.val)
    original_rgb = np.stack([image[slice_index]]*3, axis=-1) * 255
    original_rgb = original_rgb.astype(np.uint8)

    overlay = original_rgb.copy()
    for label in unique_labels:
        if label == 0:  # Skip the background
            continue
        overlay[boundaries[slice_index] & (cleaned_labels[slice_index] == label)] = color_map[label]

    ax.clear()
    ax.imshow(overlay)
    ax.set_title(f"Z-Slice {slice_index + 1}")
    ax.axis('off')
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()
