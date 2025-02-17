import logging
import numpy as np
import tifffile
from cellpose import models
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import skimage.transform
import cv2
import napari
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Load image
logging.info("Loading image...")
masks = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z_segmented_unstitched.tiff')


# Obtain image properties
print(f"Image shape: {masks.shape} in ZCYX")
total_z = masks.shape[0]
y_pixels = masks.shape[1]
x_pixels = masks.shape[2]

def stitch_by_iou(masks):
    iou_threshold = 0.5
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
    
stitched_by_iou = stitch_by_iou(masks)




# Save the labeled masks to a TIFF file
#tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_gastruloid_z.tiff', stitched_masks.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(stitched_by_iou, name='Stitched by iou')
napari.run()
