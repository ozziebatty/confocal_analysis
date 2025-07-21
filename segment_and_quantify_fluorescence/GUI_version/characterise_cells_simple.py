import os
import sys
import numpy as np
import tifffile
import csv
import pandas as pd


image_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_1\replicate_1_preprocessed.tiff")
segmented_image_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_1\replicate_1_segmentation.tiff")

channel_names = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']

image = tifffile.imread(image_path)
segmented_image = tifffile.imread(segmented_image_path)

total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

total_cells = np.max(segmented_image) + 1 #Include a background for indexing (so label 1 at position 1)

characterised_cells = {cell: {'pixel_count': 0, 'z_position': 0.0, **{channel: 0.0 for channel in channel_names}} for cell in range(total_cells)}

def quantify_cell_fluorescence(image, segmented_image):
    #Initialise arrays
    channel_values = np.zeros(total_cells, dtype=[('cell_number', int), ('pixel_count', int)] + [(channel, float) for channel in channel_names])
    z_slice_fluorescence = np.zeros(total_z, dtype=[('channels', float, total_channels), ('pixel_count', int)])
    
    for z in range(total_z):
        z_slice_image = image[z]
        z_slice_segmented_image = segmented_image[z]
        total_cells_in_z_slice = len(np.unique(z_slice_segmented_image))

        for label in np.unique(z_slice_segmented_image):
            if label == 0:
                continue

            channel_values[label]['cell_number'] = label           

            #Mask cell
            masked_cell = (z_slice_segmented_image == label)
            
            # Calculate sum of channel intensities for the cell
            channel_sums = [np.sum(z_slice_image[channel][masked_cell]) for channel in range(total_channels)]
            
            # Calculate pixel count for the cell
            running_pixel_count = np.sum(masked_cell) 
            
            # If this cell has already been partially processed in a previous slice, accumulate data
            for channel, channel_sum in zip(channel_names, channel_sums):
                channel_values[label][channel] += channel_sum
            channel_values[label]['pixel_count'] += running_pixel_count
            z_slice_fluorescence[z]['channels'] += np.array(channel_sums)
            z_slice_fluorescence[z]['pixel_count'] += running_pixel_count

    #Average channel intensities by pixel count
    for cell_data in channel_values:
        cell_label = cell_data['cell_number']
        pixel_count = cell_data['pixel_count']
        if pixel_count > 0:
            characterised_cells[cell_label]['pixel_count'] = pixel_count
            for channel in channel_names:
                average_channel_value = cell_data[channel] / pixel_count
                characterised_cells[cell_label][channel] = average_channel_value

    for z_slice in z_slice_fluorescence:
        z_slice['channels'] /= z_slice['pixel_count']

    return z_slice_fluorescence

def find_average_z_slice_of_each_label(segmented_image):
    z_slice_averages = np.zeros(total_cells, dtype=[('running_z_total', int), ('z_stack_count', int), ('average_z', float)])

    for z in range(total_z):
        z_slice_segmented_image = segmented_image[z]
        total_cells_in_z_slice = len(np.unique(z_slice_segmented_image))

        for label in np.unique(z_slice_segmented_image):
            z_slice_averages[label]['running_z_total'] += z
            z_slice_averages[label]['z_stack_count'] += 1

    for label in range(1, total_cells):
        z_slice_averages[label]['average_z'] = z_slice_averages[label]['running_z_total'] / z_slice_averages[label]['z_stack_count']

    for label in range(len(z_slice_averages)):
        characterised_cells[label]['z_position'] = z_slice_averages[label]['average_z']

def save_characterised_cells_to_csv(characterised_cells, file_path):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(characterised_cells, orient='index')

    # Ensure that the index is reset (optional, for better formatting)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cell_number'}, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

z_slice_averages = find_average_z_slice_of_each_label(segmented_image)

normalised_channel_values, z_slice_fluorescence = quantify_cell_fluorescence(image, segmented_image)

csv_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_1\replicate_1_characterised_cells.csv")

save_characterised_cells_to_csv(characterised_cells, csv_path)