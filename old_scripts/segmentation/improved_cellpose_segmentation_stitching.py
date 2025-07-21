import logging
import numpy as np
import tifffile
from cellpose import models
import napari
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Read an image from file
logging.info("Reading image...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')

# Check the shape of the image
logging.info(f"Image shape: {image.shape}")

# Initialize the Cellpose model for 2D segmentation
logging.info("Initializing Cellpose model...")
model = models.Cellpose(gpu=False, model_type='cyto2')  # Use 'cyto2' for segmentation

# Initialize an empty array for stitched masks
# Shape should be (11, 80, 80) to match the z-slices and slice dimensions
stitched_masks = np.zeros((image.shape[0], image.shape[2], image.shape[3]), dtype=np.int32)

# Perform 2D segmentation on each z-slice
logging.info("Segmenting each z-slice and stitching results...")
for z in range(image.shape[0]):
    logging.info(f"Segmenting slice {z+1}/{image.shape[0]}")
    # Extract the z-slice from all channels (assuming we are working with a single channel for segmentation)
    z_slice = image[z, 0, :, :]  # Here we use the first channel
    equalized_slice = img_as_ubyte(equalize_adapthist(z_slice, kernel_size=None, clip_limit=0.01, nbins=256))

    # Perform segmentation
    masks, flows, styles, diams = model.eval(
        equalized_slice,  # Pass a single 2D slice
        diameter=8.6,  # Automatically estimate diameter
        flow_threshold=0.4,  # Adjust as needed
        cellprob_threshold=0.0,  # Adjust as needed
        channels=[0, 0],  # Specify the channels to use (in this case, grayscale)
    )
    
    # Check the shape of masks
    logging.info(f"Shape of masks for slice {z+1}: {masks.shape}")
    
    # Ensure masks have the correct shape before assignment
    if masks.shape != stitched_masks[z, :, :].shape:
        logging.error(f"Shape mismatch: masks shape {masks.shape} does not match stitched_masks slice shape {stitched_masks[z, :, :].shape}")
        raise ValueError("Shape mismatch between masks and stitched_masks")

    # Assign masks directly to stitched_masks
    stitched_masks[z, :, :] = masks

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def relabel_stitched_masks(stitched_masks, iou_threshold=0.4):
    """Relabels the stitched 2D segmentation masks based on IoU across z-slices."""
    stitched_masks = np.squeeze(stitched_masks)
    logging.info(f"Shape of stitched_masks before relabeling: {stitched_masks.shape}")
    
    if stitched_masks.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {stitched_masks.shape}")
        
    z_dim, y_dim, x_dim = stitched_masks.shape
    current_label = 1

    # Iterate over the z-slices
    for z in range(1, z_dim):
        prev_slice = stitched_masks[z-1]
        curr_slice = stitched_masks[z]
        
        # Create a copy of the current slice to store new labels
        new_labels = np.zeros_like(curr_slice)
        
        # Find the unique labels in the current slice
        unique_labels = np.unique(curr_slice)
        
        for label in unique_labels:
            if label == 0:
                continue  # Skip background

            # Extract the current cell in the current slice
            curr_cell = curr_slice == label
            
            # Check for overlap with any cell in the previous slice
            max_iou = 0
            best_match_label = 0
            overlap_labels = np.unique(prev_slice[curr_cell])
            overlap_labels = overlap_labels[overlap_labels > 0]  # Exclude background
            
            for prev_label in overlap_labels:
                prev_cell = prev_slice == prev_label
                iou = calculate_iou(curr_cell, prev_cell)
                if iou > max_iou:
                    max_iou = iou
                    best_match_label = prev_label
            
            if max_iou >= iou_threshold:
                # If the IoU is above the threshold, assign the previous label
                new_labels[curr_cell] = best_match_label
            else:
                # Otherwise, assign a new label
                new_labels[curr_cell] = current_label
                current_label += 1
        
        # Update the current slice with the new labels
        stitched_masks[z] = new_labels

    return stitched_masks

# Apply the relabeling function after segmentation
logging.info(f"Shape of stitched_masks before relabeling: {stitched_masks.shape}")
stitched_masks = relabel_stitched_masks(stitched_masks)

# Save the labeled masks to a TIFF file
output_file = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_semi_cropped.tiff'
tifffile.imwrite(output_file, stitched_masks.astype(np.uint16))

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(image[:, 0, :, :], name='Original Image')
viewer.add_labels(stitched_masks, name='Stitched Segmentation Masks')
napari.run()
