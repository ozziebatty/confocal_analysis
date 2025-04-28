from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import tifffile
from skimage import io
import numpy as np
import pandas as pd
import napari
from tifffile import TiffFile

viewer = napari.Viewer()

# Read an image from file
original_image_pathway = '/Users/oskar/Desktop/format_test/SBSO_stellaris.tiff'

cropped_image_pathway = '/Users/oskar/Desktop/format_test/SBSO_stellaris_cropped.tiff'

image = tifffile.imread(original_image_pathway)

print(f"{datetime.now():%H:%M:%S} - Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

total_z = image.shape[0]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

# Rearrange the dimensions to (depth, height, width, channels)
rearranged_image = np.transpose(image, (0, 2, 3, 1))
#rearranged_image = np.transpose(image, (0, 2, 1))

#print("Transposed image shape:", image.shape)

# Define the ROI (Region of Interest)
z_start, z_end = 10, 15   # Example slices
y_start, y_end = 800, 900 #0 at the top
x_start, x_end = 500, 600

# Crop the image to the ROI
cropped_image = image[z_start:z_end, :, y_start:y_end, x_start:x_end]
print("Cropped image shape:", cropped_image.shape)

tifffile.imwrite(cropped_image_pathway, cropped_image)

#tifffile.imwrite(p2_pathway, p2_image)
#tifffile.imwrite(p3_pathway, p3_image)


viewer.add_image(image)
viewer.add_image(cropped_image)
napari.run()

