
#%%
import numpy as np
import cv2
import tifffile
from datetime import datetime
from cellpose import models
import napari

dapi_channel = 0
cell_diameter = 8.4
flow_threshold = 1.0
cellprob_threshold = -10
iou_threshold = 0.5  # IoU threshold for stitching

input_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris.tiff"  # Path to your input image
unstitched_output_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris_segmented_unstitched.tiff"  # Path to save the output image
stitched_output_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris_segmented_stitched.tiff"  # Path to save the stitched image
#%%

def segment_stitch_and_clean(channel_slice):    # Unpack parameters

    def segment_2D(channel_slice):
        """Segment nuclei in 2D slices using Cellpose"""
        model = models.Cellpose(gpu=False, model_type='nuclei')
        
        total_z = channel_slice.shape[0]
        segmented_image = np.zeros_like(channel_slice, dtype=np.uint32)
        total_cells_segmented = 0
        
        for z in range(total_z):
            print(f"{datetime.now():%H:%M:%S} - Segmenting z-slice {z + 1}/{total_z}")
            
            z_slice = channel_slice[z]
            segmented_image_z_slice, flows, styles, diams = model.eval(
                z_slice,
                diameter=cell_diameter,
                flow_threshold=flow_threshold,  # Higher is more cells
                cellprob_threshold=cellprob_threshold,  # Lower is more cells
            )
        
            segmented_image[z] = segmented_image_z_slice
            total_cells_segmented += len(np.unique(segmented_image_z_slice)) - 1  # Subtract 1 for background (0)
        
        print(f"Total cells segmented (unstitched): {total_cells_segmented}")
        print(f"Cell diameter = {cell_diameter}, Flow threshold = {flow_threshold}")
                      
        return segmented_image

    def stitch_by_iou(segmented_unstitched):
        
        def calculate_iou(cell_1, cell_2):
            """Calculate Intersection over Union (IoU) between two binary masks"""
            intersection = np.logical_and(cell_1, cell_2).sum()
            union = np.logical_or(cell_1, cell_2).sum()
            if union == 0:
                return 0
            return intersection / union
        
        """Stitch segmented cells across z-slices based on IoU"""
        total_z = segmented_unstitched.shape[0]
        
        def relabel_stitched_masks(segmented_unstitched):
            """Relabels the stitched 2D segmentation masks based on IoU across z-slices"""
            stitched_image = np.squeeze(segmented_unstitched).astype(np.uint32)
                
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

            total_cells_stitched = current_label
        
            print(f"Total cells segmented (stitched): {total_cells_stitched}")


            return stitched_image

        relabelled_stitched_masks = relabel_stitched_masks(segmented_unstitched)

        return relabelled_stitched_masks
    
    
    def clean_labels(segmented_stitched):
        # This function relabels the segmented image so that no labels are skipped.
        print(f"{datetime.now():%H:%M:%S} - Cleaning labels...")

        # Get the unique values in the segmented image, excluding 0 (background, if applicable)
        unique_labels = np.unique(segmented_stitched)
        
        # Create a mapping from old labels to new labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Relabel the segmented image using the mapping
        relabeled_image = np.vectorize(label_mapping.get)(segmented_stitched)
        
        total_clean_cells_stitched = len(np.unique(relabeled_image))
        
        print(f"Total cells segmented (cleaned): {total_clean_cells_stitched}")

        
        return relabeled_image

    #segmented_unstitched = segment_2D(channel_slice)

    #tifffile.imwrite(unstitched_output_path, segmented_unstitched.astype(np.uint32))  # Save the segmented image
    #print(f"Segmented image saved to {unstitched_output_path}")

    segmented_unstitched = tifffile.imread(unstitched_output_path)  # Load your image here


    segmented_stitched = stitch_by_iou(segmented_unstitched)
    
    tifffile.imwrite(stitched_output_path, segmented_stitched.astype(np.uint32))  # Save the segmented image
    print(f"Stitched Segmented image saved to {stitched_output_path}")

    cleaned_segmented_stitched = clean_labels(segmented_stitched)
    
    tifffile.imwrite(stitched_output_path, cleaned_segmented_stitched.astype(np.uint32))  # Save the segmented image
    print(f"Cleaned Stitched Segmented image saved to {stitched_output_path}")    
    
    return cleaned_segmented_stitched
    
#%%

image = tifffile.imread(input_path)  # Load your image here
channel_slice = image[:, 0, :, :]  # Extract the DAPI channel (0) and first z-slice (0)

#%%

segmented_image = segment_stitch_and_clean(channel_slice)  # Segment the DAPI channel

#%%
segmented_image = tifffile.imread(stitched_output_path)  # Load your image here

print("Loading napari")
viewer = napari.Viewer()
viewer.add_image(channel_slice)  # Add the segmented image to the viewer
viewer.add_labels(segmented_image)  # Add the segmented image to the viewer
napari.run()  # Start the napari viewer

# %%
