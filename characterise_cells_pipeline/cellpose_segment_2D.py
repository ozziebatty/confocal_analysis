from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import numpy as np
import tifffile
from cellpose import models
import napari
import sys

cell_diameter = 8
flow_threshold = 0.5
cellprob_threshold = 0.1

debug_mode = True

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/translation_NMPs_image_analysis/results/NM_SBSE_OPP_d5_2024_10_28__17_40_50__p5'
    dapi_channel = 2
    cpu_num = ''
    display_napari = 'True'
else:
    folder_pathway = sys.argv[1]
    cpu_num = ''#sys.argv[2]
    dapi_channel = int(sys.argv[3])
    display_napari = sys.argv[4]

image_pathway = folder_pathway + '/preprocessed' + cpu_num + '.tiff'
image = tifffile.imread(image_pathway)

print("Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

total_z = image.shape[0]

channel_to_segment = dapi_channel

channel_slice = image[:, channel_to_segment, :, :]

def segment_2D(channel_slice):
    model = models.Cellpose(gpu=False, model_type='nuclei')  # Use 'cyto2' for segmentation

    segmented_image = np.zeros_like(channel_slice, dtype=np.uint16)
    total_cells_segmented = 0

    for z in range(total_z):
        if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
            print(f"{datetime.now():%H:%M:%S} - Segmentation {((z + 1) * 100) // total_z}% complete")

        z_slice = channel_slice[z]
        segmented_image_z_slice, flows, styles, diams = model.eval(
            z_slice,
            diameter=cell_diameter,
            flow_threshold=0.5, #Higher is more cells.
            cellprob_threshold=0.1, #Lower is more cells
        )
    
        segmented_image[z] = segmented_image_z_slice
        total_cells_segmented += len(np.unique(segmented_image_z_slice))
    

    #print("Cells segmented:", total_cells_segmented)
    #print("Cell diameter =", cell_diameter, ", Flow threshold =", flow_threshold, ", Cellprob threshold =", cellprob_threshold, ", Dapi Channel =", dapi_channel)


    return segmented_image

def display(image, segmented_image):

    if display_napari == 'True':
        import napari
        viewer = napari.Viewer()
        viewer.add_image(channel_slice)
        viewer.add_labels(segmented_image)
        napari.run()

def save(image, folder_pathway):
    output_pathway = folder_pathway + '/segmented' + cpu_num + '.tiff'
    tifffile.imwrite(output_pathway, image)

segmented_image = segment_2D(channel_slice)

save(segmented_image, folder_pathway)

display(image, segmented_image)

print("Segmentation succesfully complete")