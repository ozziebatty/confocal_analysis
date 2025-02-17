import logging
import numpy as np
import tifffile
import cv2
import napari
import csv
import pandas as pd

viewer = napari.Viewer()


np.set_printoptions(threshold = np.inf)

cropping_degree = 'very_cropped'
cell_diameter = 8.6

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting image processing...")

#Load image
logging.info("Loading image...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/segmented_very_cropped.tiff')

# Obtain image properties
print(f"Image shape: {image.shape} in ZCYX")
total_z = image.shape[0]
y_pixels = image.shape[1]
x_pixels = image.shape[2]
max_label = np.max(image)

labeled_image = np.zeros_like(image)
cell_A_image = np.zeros_like(image)
cell_B_image = np.zeros_like(image)

stitching_threshold = 0.5
size_importance = 0.3

label_properties = []
label_overlaps = np.zeros((total_z - 1, max_label + 1, max_label + 1), dtype=int)
stitching_probabilities = np.zeros_like(label_overlaps)
label_positions = {}

stitching_thresholds = []

# Calculate label_properties and label_overlaps
for z in range(total_z):

    z_slice = image[z]
    if z < total_z - 1:
        next_z_slice = image[z+1]

    # Get unique labels and their counts in the current z-slice
    labels, counts = np.unique(z_slice, return_counts=True)
    unique_labels = np.max(labels)
    properties = [None] * (unique_labels + 1)
    
    # Dictionary to keep track of x and y positions for each label
    label_positions = {label: {"x_sum": 0, "y_sum": 0, "pixel_count": 0} for label in labels if label != 0}

    # Go through each pixel in the current z-slice
    for x in range(x_pixels):
        for y in range(y_pixels):
            label = z_slice[x, y]

            if label != 0:  # Ignore background (label 0)
                # Update pixel count, x and y positions for this label
                label_positions[label]["pixel_count"] += 1
                label_positions[label]["x_sum"] += x
                label_positions[label]["y_sum"] += y


                # If not the last z-slice, check the next z-slice for overlaps
                if z < total_z - 1:
                    next_z_label = next_z_slice[x, y]
                    #print("next_z_label", next_z_label)

                    if next_z_label != 0:
                        # Update label_overlaps matrix

                        label_overlaps[z][label, next_z_label] += 1

                              
    # At the end of the slice, calculate the center positions for each label
    for label, pos_data in label_positions.items():
        if pos_data["pixel_count"] > 0:
            x_center = pos_data["x_sum"] / pos_data["pixel_count"]
            y_center = pos_data["y_sum"] / pos_data["pixel_count"]
##            properties.append({
##                "label": label,
##                "pixel_count": pos_data["pixel_count"],
##                "x_position": x_center,
##                "y_position": y_center
##            })

            #print(label)
            properties[label] = {
                "label": label,
                "pixel_count": pos_data["pixel_count"],
                "x_position": x_center,
                "y_position": y_center
            }

    # Store the label properties for the current z-slice
    label_properties.append(properties)

#Calculate probabilities that this label should be stitched with each label in z + 1
for z in range(total_z - 1):
    z_slice_label_properties = label_properties[z][1:]
    next_z_slice_label_properties = label_properties[z + 1][1:]
    z_slice_label_overlaps = label_overlaps[z]
    
    for each_label in z_slice_label_properties:
        label = each_label["label"]
        label_pixel_count = each_label["pixel_count"]
        label_x_position = each_label["x_position"]
        label_y_position = each_label["y_position"]
        
        #Candidates where overlapping pixels with z+1 > 0
        candidates = np.where(z_slice_label_overlaps[label] > 0)[0]
        overlap_values = z_slice_label_overlaps[label][candidates]
        
        for candidate in candidates:
            candidate_pixel_count = next_z_slice_label_properties[candidate - 1]["pixel_count"]
            overlap_value = z_slice_label_overlaps[label, candidate]
            if candidate_pixel_count > label_pixel_count:
                smaller_cell = label_pixel_count
                larger_cell = candidate_pixel_count
            else:
                smaller_cell = candidate_pixel_count
                larger_cell = label_pixel_count
                
            #print(next_z_slice_label_properties[candidate])
            proportional_overlap = overlap_value/smaller_cell
            required_threshold = stitching_threshold * (larger_cell ** (-size_importance)) * 12
            above_threshold = proportional_overlap - required_threshold
            if above_threshold > 0:
                stitching_probabilities[z][label, candidate] = above_threshold
                stitching_thresholds.append([above_threshold, z, label, candidate, False])
            
                #print("label", label, "matches with candidate", candidate)

    stitching_thresholds.sort(key=lambda x: x[0], reverse=True)

new_label_number = 0
for pair in stitching_thresholds:
    if pair[4] == False: #if paired = False
        #print(pair)
        z = pair[1]
        label_A = pair[2]
        label_B = pair[3]
        #print("New Pair!")
        #print(label)
        #labeled_image[label
        cell_A = image[z] == label_A
        cell_B = image[z+1] == label_B
        #print(np.unique(cell_to_relabel))
        labeled_image[z][cell_A] = new_label_number
        cell_A_image[z][cell_A] = new_label_number
        cell_B_image[z+1][cell_B] = new_label_number
        labeled_image[z+1][cell_B] = new_label_number
        new_label_number += 1

        if new_label_number == 4:
            viewer.add_labels(cell_A_image)
            viewer.add_labels(cell_B_image)

##
##            print(label_A)
##            print(label_B)
##            print(cell_A)
##            print(cell_B)
            #print(labeled_image)



        #Assign same labels to pair        
        
        for other_pair in stitching_thresholds:
            if other_pair[1] == z and other_pair[2] == label:
                other_pair[4] = True

#print(stitching_thresholds)
    #removes items that have same z and same label
        #print(above_threshold)
        #print(proportional_overlap)


logging.info("COMPLETE")

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

#stitched_masks = stitch_2D_slices(masks)

# Save the labeled masks to a TIFF file
#tifffile.imwrite((file_pathways.loc[cropping_degree, 'segmented_tiff']), stitched_masks.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
napari.run()
