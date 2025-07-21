from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import numpy as np
import tifffile
import sys

iou_threshold = 0.5

debug_mode = False

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images'
    dapi_channel = 3
    display_napari = 'True'
else:
    folder_pathway = sys.argv[1]
    cpu_num = ''#sys.argv[2]
    dapi_channel = int(sys.argv[3])
    display_napari = sys.argv[4]

display_napari = 'True'

segmented_image_pathway = folder_pathway + '/segmented' + cpu_num + '.tiff'
segmented_image = tifffile.imread(segmented_image_pathway)

#print(f"{datetime.now():%H:%M:%S} - Segmented image shape (z, y, x) :", segmented_image.shape, "dtype:", segmented_image.dtype)

total_z = segmented_image.shape[0]


total_unstitched_cells = 0
for z in range(total_z):
    total_unstitched_cells += len(np.unique(segmented_image[z]))

#print(f"{datetime.now():%H:%M:%S} - Unstitched cells =", total_unstitched_cells)

def stitch_by_iou(segmented_image):

    def calculate_iou(cell_1, cell_2):
        #Calculate Intersection over Union (IoU) between two binary masks
        intersection = np.logical_and(cell_1, cell_2).sum()
        union = np.logical_or(cell_1, cell_2).sum()
        if union == 0:
            return 0
        return intersection / union

    def relabel_stitched_masks(segmented_image):
        #Relabels the stitched 2D segmentation masks based on IoU across z-slices
        stitched_image = np.squeeze(segmented_image).astype(np.uint16)
         
        current_label = 1
        for z in range(1, total_z):
            if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
                print(f"{datetime.now():%H:%M:%S} - Stitching {((z + 1) * 100) // total_z}% complete)")

            previous_slice = stitched_image[z-1]
            current_slice = stitched_image[z]
            
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
            stitched_image[z] = new_labels

        return stitched_image

    relabelled_stitched_masks = relabel_stitched_masks(segmented_image)

    return relabelled_stitched_masks

def clean_labels(stitched_image):
    #Relabels segmented image so that every cell has a unique label and none are skipped
    print(f"{datetime.now():%H:%M:%S} - Cleaning labels...")

    # Get the unique values in the segmented image, excluding 0 (background, if applicable)
    unique_labels = np.unique(stitched_image)
    
    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Relabel the segmented image using the mapping
    relabeled_image = np.vectorize(label_mapping.get)(stitched_image)
    
    return relabeled_image

stitched_image = stitch_by_iou(segmented_image)

cleaned_stitched_image = clean_labels(stitched_image)

#print("Stitched cells:", (len(np.unique(stitched_image)) - 1)) #Minus background

def display(stitched_image):
    #print(f"{datetime.now():%H:%M:%S} - Displaying...")
    if display_napari == 'True':
        import napari

        original_image_pathway = folder_pathway + '/preprocessed.tiff'
        original_image = tifffile.imread(original_image_pathway)

        dapi_channel_slice = original_image[:, dapi_channel, :, :]
    
        viewer = napari.Viewer()
        viewer.add_image(dapi_channel_slice)
        viewer.add_labels(stitched_image)
        napari.run()

def save(image, folder_pathway):
    new_pathway = folder_pathway + '/stitched' + cpu_num + '.tiff'
    tifffile.imwrite(new_pathway, image)

save(cleaned_stitched_image, folder_pathway)

display(cleaned_stitched_image)

print("Stitching succesfully complete")