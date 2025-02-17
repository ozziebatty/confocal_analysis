print("Importing...")

import tifffile
import numpy as np
import napari
from tifffile import TiffFile

viewer = napari.Viewer()

# Read an image from file
image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSO_example_image_combined.tiff'
top_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSO_example_image_1.tiff'
bottom_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSO_example_image_2.tiff'

image = tifffile.imread(image_pathway)

print("Original image shape:", image.shape)

# Crop the image to the ROI
top_image = image[0]
bottom_image = image[1]
print("Top image shape:", top_image.shape)
print("Bottom image shape:", bottom_image.shape)

tifffile.imwrite(top_image_pathway, top_image)
tifffile.imwrite(bottom_image_pathway, bottom_image)

print("Displaying")
viewer.add_image(top_image)
viewer.add_image(bottom_image)

napari.run()

