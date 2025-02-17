from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages for characterise_cells...")

import sys
import numpy as np
import tifffile
import napari
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.odr import ODR, Model, Data
from sklearn.linear_model import LinearRegression
import ast
import os

debug_mode = False
condition = 'control'

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_code_test/results/SBSE'
    threshold_folder = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_code_test/results'
    cpu_num = ''
    channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
    dapi_channel = 3
    display_napari = 'False'
else:
    folder_pathway = sys.argv[1]
    threshold_folder = sys.argv[2]
    cpu_num = ''#sys.argv[3]
    channel_names = ast.literal_eval(sys.argv[4])
    dapi_channel = int(sys.argv[5])
    display_napari = sys.argv[6]

thresholds =  {'Sox1': [0.12976700063980764, 0.07381177417068815, 27.47377505150176],
               'Sox2 Cyan': [0.0735035995708027, 0.08741172789897643, 18.596481770556615],
               'Sox2 Orange': [0.04518975473155211, 0.006305888409557325, 5.1160452844525395],
               'Bra': [0.0808175471629982, 0.054052924114643686, 18.137781466492786]}

remove_channel = 'Sox2 Orange'

channel_colours = ['green', 'blue', 'orange', 'grey', 'red']
display_colours = {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'grey': [100, 100, 100],
    'purple': [150, 0, 150],
    'orange': [150, 150, 0]
}

#print(f"{datetime.now():%H:%M:%S} - Loading images...")
image_pathway = folder_pathway + '/deconvoluted' + cpu_num + '.tiff'
segmented_image_pathway = folder_pathway + '/stitched' + cpu_num + '.tiff'
image = tifffile.imread(image_pathway)
segmented_image = tifffile.imread(segmented_image_pathway)

#print(f"{datetime.now():%H:%M:%S} - Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)
#print(f"{datetime.now():%H:%M:%S} - Segmented image shape (z, c, y, x) :", segmented_image.shape, "dtype:", segmented_image.dtype)

total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]


print(total_cells , " cells to characterise")

characterised_cells = {cell: {'pixel_count': 0, 'z_position': 0.0, 'fate': '', 'display_colour': [0, 0, 0], **{channel: 0.0 for channel in channel_names}} for cell in range(total_cells)}
characterised_cells_summary = {}

def quantify_cell_fluorescence(image, segmented_image):
    #print(f"{datetime.now():%H:%M:%S} - Quantifying cell fluorescence...")

    #Initialise arrays
    #print("total_cells", total_cells)
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


    #Normalise cell intensities relative to DAPI
    #print(f"{datetime.now():%H:%M:%S} - Normalising intensities to DAPI...")
    normalised_channel_values = np.zeros(total_cells, dtype=[('cell_number', int)] + [(channel, float) for channel in channel_names])
    for cell_data in channel_values:
        cell_label = cell_data['cell_number']
        normalised_channel_values[cell_label]['cell_number'] = cell_label

        DAPI_value = cell_data['DAPI']

        for channel in channel_names:
            channel_value = cell_data[channel]
            if DAPI_value > 0:
                normalised_channel_value = channel_value / DAPI_value
                if normalised_channel_value > 0:
                    normalised_channel_values[cell_label][channel] = normalised_channel_value
            else:
                normalised_channel_values[cell_label][channel] = 0

    return normalised_channel_values

def find_average_z_slice_of_each_label(segmented_image):
    print(f"{datetime.now():%H:%M:%S} - Calculating z positions...")
    
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

def assign_fates_by_intensity_to_threshold(characterised_cells):
    #print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    characterised_cells_list = list(characterised_cells.items())

    for channel in channel_names:
        if not channel == 'DAPI':
            if not channel == remove_channel:
                a, b, c = thresholds[channel]

                for cell in characterised_cells_list:
                    cell_label = cell[0]
                    DAPI_fluorescence = cell[1]['DAPI']
                    channel_fluorescence = cell[1][channel]
                    z_position = cell[1]['z_position']

                    threshold = a * DAPI_fluorescence + b * z_position + c

                    if channel_fluorescence > threshold:
                        characterised_cells[cell_label]['fate'] += channel + '+, '
                        original_colour = characterised_cells[cell_label]['display_colour']
                        new_colour = display_colours[channel_colours[channel_names.index(channel)]]
                        blended_colour = np.add(original_colour, new_colour).tolist()
                        characterised_cells[cell_label]['display_colour'] = blended_colour

    #Assign Background to label 0
    characterised_cells[0]['fate'] = 'Background'

    #If not assigned, assign 'Unlabelled'
    for cell_label in range(1, len(characterised_cells_list)):
        if characterised_cells[cell_label]['fate'] == '':  # Check if the fate string is empty
            characterised_cells[cell_label]['fate'] = 'Unlabelled'
            characterised_cells[cell_label]['display_colour'] = display_colours['grey']

    #Count fates
    # Extract all fates into a list
    fates = [cell_data['fate'] for cell_data in characterised_cells.values()]
    fate_counts = Counter(fates)

    for fate, count in fate_counts.items():
        if not fate == 'Background':
            characterised_cells_summary[fate] = count
            #print(f"{fate}: {count}")

    print(characterised_cells_summary)

def save_characterised_cells_to_csv(characterised_cells, folder_pathway):
    file_pathway = folder_pathway + '/characterised_cells' + cpu_num + '.csv'

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(characterised_cells, orient='index')

    # Ensure that the index is reset (optional, for better formatting)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cell_number'}, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_pathway, index=False)

def save_characterised_cells_summary_to_csv(charactarised_cells_summary, folder_pathway):
    image_name = folder_pathway[len(threshold_folder):][1:]

    # File path for the CSV file
    csv_pathway = threshold_folder + "/characterised_cells_summary.csv"

    #Ensure the CSV file exists and create it if not
    if not os.path.exists(csv_pathway):
        with open(csv_pathway, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['image_name', 'condition'] + list(characterised_cells_summary.keys()))
            writer.writeheader()

    # Read the existing file to determine missing columns
    with open(csv_pathway, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        existing_columns = reader.fieldnames

    # Add missing columns if needed
    missing_columns = [fate for fate in characterised_cells_summary.keys() if fate not in existing_columns]
    if missing_columns:
        with open(csv_pathway, mode='r', newline='') as file:
            rows = list(csv.DictReader(file))

        with open(csv_pathway, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=existing_columns + missing_columns)
            writer.writeheader()
            writer.writerows(rows)

    # Append the new row
    with open(csv_pathway, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=existing_columns + missing_columns)
        row = {'image_name': image_name, 'condition': condition, **characterised_cells_summary}
        writer.writerow(row)

def create_characterised_image(characterised_cells):
    print(f"{datetime.now():%H:%M:%S} - Creating characterised image...")

    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)

    for cell, data, in characterised_cells.items():
        if cell%1000 == 0:
            print("Progress - ", cell, "/", total_cells)
        display_colour = data['display_colour']

        characterised_image[segmented_image == cell] = display_colour

    # Visualize with Napari
    print(f"{datetime.now():%H:%M:%S} - Visualizing results with Napari...")
    viewer = napari.Viewer()
    viewer.add_image(characterised_image)
    viewer.add_image(image)
    napari.run()

    return characterised_image


z_slice_averages = find_average_z_slice_of_each_label(segmented_image)

normalised_channel_values = quantify_cell_fluorescence(image, segmented_image)

assign_fates_by_intensity_to_threshold(characterised_cells)

save_characterised_cells_to_csv(characterised_cells, folder_pathway)

save_characterised_cells_summary_to_csv(characterised_cells_summary, folder_pathway)

#characterised_image = create_characterised_image(characterised_cells)

print(f"{datetime.now():%H:%M:%S} - Cell characterisation succesfully complete")
