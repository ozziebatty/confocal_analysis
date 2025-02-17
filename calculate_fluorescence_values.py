import numpy as np
import tifffile
import napari
import csv
import pandas as pd
import sys
import os

debug_mode = False

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_very_cropped'
    dapi_channel = 0
    print(dapi_channel)
    print(type(dapi_channel))
    display_napari = 'False'
else:
    folder_pathway = sys.argv[1]
    dapi_channel = int(sys.argv[2])
    display_napari = sys.argv[3]


image_pathway = folder_pathway + '/preprocessed.tiff'
image = tifffile.imread(image_pathway)

segmented_image_pathway = folder_pathway + '/stitched.tiff'
segmented_image = tifffile.imread(segmented_image_pathway)

#Image properties prior to resizing
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

total_cells =len(np.unique(segmented_image))

print("Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)
print("Stitched image shape (z, y, x) :", segmented_image.shape, "dtype:", image.dtype)
print(total_cells - 1, " cells to characterise") #Exclude the background

#Initialise characterised_cells
characterised_cells = {cell: {'pixel_count': 0, 'fate': 'Unlabelled'} for cell in range(1, total_cells + 1)}


def quantify_cell_fluorescence(image, segmented_image):
    #Initialise arrays
    cell_fluorescence = np.zeros(total_cells, dtype=[('cell_number', int), ('channels', float, total_channels), ('pixel_count', int)])
    z_slice_fluorescence = np.zeros(total_z, dtype=[('channels', float, total_channels), ('pixel_count', int)])
    
    for z in range(total_z):
        z_slice_image = image[z]
        z_slice_segmented_image = segmented_image[z]

        for label in np.unique(z_slice_segmented_image):
            cell_fluorescence[label]['cell_number'] = label           
            if label == 0:  # Skip background
                continue
                
            #Mask cell
            masked_cell = (z_slice_segmented_image == label)
            
            # Calculate sum of channel intensities for the cell
            channel_sums = [np.sum(z_slice_image[channel][masked_cell]) for channel in range(total_channels)]
            
            # Calculate pixel count for the cell
            running_pixel_count = np.sum(masked_cell)
            
            # If this cell has already been partially processed in a previous slice, accumulate data
            cell_fluorescence[label]['channels'] += np.array(channel_sums)
            cell_fluorescence[label]['pixel_count'] += running_pixel_count
            z_slice_fluorescence[z]['channels'] += np.array(channel_sums)
            z_slice_fluorescence[z]['pixel_count'] += running_pixel_count

    #Average channel intensities by pixel count
    for cell in cell_fluorescence:
        cell_number = cell[0]
        pixel_count = cell['pixel_count']
        if pixel_count > 0:
            cell['channels'] /= pixel_count
            characterised_cells[cell_number]['pixel_count'] = pixel_count
            channels = cell['channels']
            for channel in range(total_channels):
                characterised_cells[cell_number][f'channel_{channel}'] = channels[channel]

    for z_slice in z_slice_fluorescence:
        z_slice['channels'] /= z_slice['pixel_count']

    #Normalise cell intensities relative to DAPI
    normalised_cell_fluorescence = np.zeros(len(cell_fluorescence), dtype=[('cell_number', int), ('channels', float, total_channels)])
    for cell in cell_fluorescence:
        
        cell_number = cell['cell_number']
        normal_value = cell['channels'][dapi_channel]
        
        if normal_value > 0:
            normalised_channels = cell['channels'] / normal_value
        else:
            normalised_channels = np.zeros_like(cell['channels'])
            
        normalised_cell_fluorescence[cell_number]['cell_number'] = cell_number
        normalised_cell_fluorescence[cell_number]['channels'] = normalised_channels
    
    return normalised_cell_fluorescence, z_slice_fluorescence

def save_to_csv():
    print("Saving data...")
    
    characterised_cells_pathway = folder_pathway + '/characterised_cells.csv'
    characterised_cells_df = pd.DataFrame.from_dict(characterised_cells, orient='index')
    characterised_cells_df.reset_index(inplace=True)
    characterised_cells_df.rename(columns={'index': 'cell_number'}, inplace=True)

    normalised_cell_fluorescence_df = pd.DataFrame({
    'cell': normalised_cell_fluorescence['cell_number'],
    'channel_0': normalised_cell_fluorescence['channels'][:, 0],
    'channel_1': normalised_cell_fluorescence['channels'][:, 1],
    'channel_2': normalised_cell_fluorescence['channels'][:, 2],
    'channel_3': normalised_cell_fluorescence['channels'][:, 3]
    })

    z_slice_fluorescence_df = pd.DataFrame({
        'z_slice': range(1, total_z + 1),
        'channel_0': [item[0][0] for item in z_slice_fluorescence],
        'channel_1': [item[0][1] for item in z_slice_fluorescence],
        'channel_2': [item[0][2] for item in z_slice_fluorescence],
        'channel_3': [item[0][3] for item in z_slice_fluorescence],
        'pixel_count': [item[1] for item in z_slice_fluorescence]
    })

    
    characterised_cells_df.to_csv(characterised_cells_pathway, index=False)
    normalised_cell_fluorescence_df.to_csv(normalised_cell_fluorescence_pathway, index=False)
    z_slice_fluorescence_df.to_csv(z_slice_fluorescence_pathway, index=False)

normalised_cell_fluorescence, z_slice_fluorescence = quantify_cell_fluorescence(image, segmented_image)

characterised_cells_pathway = folder_pathway + '/characterised_cells.csv'
normalised_cell_fluorescence_pathway = folder_pathway + '/normalised_cell_fluorescence.csv'
z_slice_fluorescence_pathway = folder_pathway + '/z_slice_fluorescence.csv'

save_to_csv()


