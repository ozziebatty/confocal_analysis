from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - Importing packages...")

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

cropping_degree = 'tall'
scale_factor = 1
cell_diameter = 8.6

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
zoom_factor_z = scale_factor*0.9/1.25
resampled_image = zoom(image, (zoom_factor_z, 1, 1, 1), order=3)  # Cubic interpolation
image = resampled_image

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

    resized_image = np.zeros((total_z, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.float32)
    
    for z in range(total_z):
        z_slice = image[z][channel_to_segment]

        resized_z_slice = skimage.transform.resize(z_slice, (y_pixels * scale_factor, x_pixels * scale_factor), anti_aliasing=True)


        #z_slice_in_process = z_slice[channel_to_segment]
        z_slice_in_process = resized_z_slice

        equalized = img_as_ubyte(equalize_adapthist(z_slice_in_process, kernel_size=None, clip_limit=0.01, nbins=256))
        equalized_twice = img_as_ubyte(equalize_adapthist(equalized, kernel_size=None, clip_limit=0.01, nbins=256))
        equalized_thrice = img_as_ubyte(equalize_adapthist(equalized_twice, kernel_size=None, clip_limit=0.01, nbins=256))

##        # Enhance the edges
##        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
##        mask = cv2.subtract(equalized, blurred)
##        mask = cv2.multiply(mask, 1.5)
##        sharpened = cv2.add(equalized, mask)
##
##        z_slice_in_progress = sharpened


        #preprocessed_image[z] = equalized_thrice

        resized_image[z] = equalized_thrice


    return resized_image

def segment_2D(image):
    # Initialize Cellpose
    logging.info("Initializing Cellpose...")
    model = models.Cellpose(gpu=False, model_type='cyto2')  # Use 'cyto2' for segmentation

    # Perform 2D segmentation on each z-slice
    masks = np.zeros((total_z, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.int32)

    for z in range(total_z):
        logging.info(f"Segmenting slice {z+1}/{total_z}")
        z_slice = image[z]

        z_slice_masks, flows, styles, diams = model.eval(
            z_slice,
            diameter=cell_diameter * scale_factor,  # Adjust as needed
            flow_threshold=0.6,  # Adjust as needed
            cellprob_threshold=0.4,  # Adjust as needed
        )
        
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
tifffile.imwrite((file_pathways.loc[cropping_degree, 'segmented_tiff']), stitched_masks.astype(np.uint16))
#tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_gastruloid_z.tiff', stitched_masks.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
#viewer.add_image(image, name='Original Image')
viewer.add_image(preprocessed_image, name='Preprocessed Image')
viewer.add_labels(stitched_masks, name='Stitched Segmentation Masks')
napari.run()
