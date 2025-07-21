import logging
import numpy as np
import tifffile
from cellpose import models
import napari
import OpenGL.GL as gl
import OpenGL.GLUT as glut

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting image processing...")

# Read an image from file
logging.info("Reading image...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/very_cropped_gastruloid_z.tiff')

# Initialize the Cellpose model for 3D
logging.info("Initializing Cellpose model...")
model = models.Cellpose(gpu=False, model_type='cyto2')  # Use 'cyto2' for 3D segmentation

# Perform segmentation with appropriate parameters
logging.info("Segmenting image...")
masks, flows, styles, diams = model.eval(
    image,
    diameter=None,  # Automatically estimate diameter
    flow_threshold=0.4,  # Adjust as needed
    cellprob_threshold=0.0,  # Adjust as needed
    z_axis=0
)

# Check the shapes
logging.info(f"Original image shape: {image.shape}")
logging.info(f"Masks shape: {masks.shape}")

# Check the shape again after adjustment
logging.info(f"Adjusted masks shape: {masks.shape}")

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(image, name='Original Image')
viewer.add_labels(masks, name='Segmentation Masks')
napari.run()

