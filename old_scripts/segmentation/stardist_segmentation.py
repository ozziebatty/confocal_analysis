import logging
import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import cv2
import napari
import stardist
from stardist.models import StarDist3D
from stardist.models import StarDist2D
print(stardist.__version__)
print(dir(stardist))
print('\n')
import json
import numpy as np
import tensorflow as tf

import logging
import numpy as np
import tifffile
from cellpose import models
import napari
from skimage.exposure import equalize_adapthist
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Load image
logging.info("Loading image...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/very_cropped_gastruloid_z.tiff')

# Obtain image properties
print(f"Image shape: {image.shape} in ZCYX")
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

# Define parameters
channel_to_segment = 0 #Uses the 1st channel (DAPI) to segment with
iou_threshold = 0.5 #Threshold given to be classed as the same label

def preprocess(image):
    logging.info("Preprocessing...")
    preprocessed_image = np.zeros((total_z, y_pixels, x_pixels), dtype=np.int32)
    
    for z in range(total_z):
        z_slice = image[z]

        z_slice_in_process = z_slice[channel_to_segment]

        #z_slice_in_process = cv2.GaussianBlur(z_slice_in_process, (1, 1), 0)
        z_slice_in_process = img_as_ubyte(equalize_adapthist(z_slice_in_process, kernel_size=None, clip_limit=0.01, nbins=256))

        preprocessed_image[z] = z_slice_in_process


    return preprocessed_image

def segment_2D(image):
    # Initialize StarDist
    logging.info("Initialising StarDist...")

    model = StarDist2D.from_pretrained('/Users/oskar/.keras/models/StarDist2D/2D_versatile_fluo')

    # Perform 2D segmentation on each z-slice
    masks = np.zeros((total_z, y_pixels, x_pixels), dtype=np.int32)

    for z in range(total_z):
        logging.info(f"Segmenting slice {z+1}/{total_z}")
        z_slice = image[z]

        labels, details = model.predict_instances(z_slice)

        
        masks[z, :, :] = z_slice_masks

    return masks

def stitch_2D_slices(masks):
    logging.info("Stitching masks...")

    def calculate_iou(mask1, mask2):
        #Calculate Intersection over Union (IoU) between two binary masks
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union

    def relabel_stitched_masks(masks):
        #Relabels the stitched 2D segmentation masks based on IoU across z-slices
        stitched_masks = np.squeeze(masks)
         
        current_label = 1
        for z in range(1, total_z):
            previous_slice = stitched_masks[z-1]
            current_slice = stitched_masks[z]
            
            # Create a copy of the current slice to store new labels
            new_labels = np.zeros_like(current_slice)
            
            # Find the unique labels in the current slice
            unique_labels = np.unique(current_slice)
            
            for label in unique_labels:
                if label == 0:
                    continue  # Skip background

                # Extract the current cell in the current slice
                current_cell = current_slice == label
                
                # Check for overlap with any cell in the previous slice
                max_iou = 0
                best_match_label = 0
                overlap_labels = np.unique(previous_slice[current_cell])
                overlap_labels = overlap_labels[overlap_labels > 0]  # Exclude background
                
                for previous_label in overlap_labels:
                    previous_cell = previous_slice == previous_label
                    iou = calculate_iou(current_cell, previous_cell)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_label = previous_label
                
                if max_iou >= iou_threshold:
                    # If the IoU is above the threshold, assign the previous label
                    new_labels[current_cell] = best_match_label
                else:
                    # Otherwise, assign a new label
                    new_labels[current_cell] = current_label
                    current_label += 1
            
            # Update the current slice with the new labels
            stitched_masks[z] = new_labels

        print("Total cells segmented: ", len(np.unique(stitched_masks)))
        return stitched_masks

    relabelled_stitched_masks = relabel_stitched_masks(masks)

    return relabelled_stitched_masks
    

preprocessed_image = preprocess(image)
masks = segment_2D(preprocessed_image)
stitched_masks = stitch_2D_slices(masks)


# Save the labeled masks to a TIFF file
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/stardist_segmented_stitching_very_cropped_gastruloid_z.tiff', stitched_masks.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(preprocessed_image, name='Original Image')
viewer.add_labels(stitched_masks, name='Stitched Segmentation Masks')
napari.run()
