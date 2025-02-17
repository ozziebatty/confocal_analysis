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

debug_mode = False

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis_thresholds'
    file_name = 'placeholder'
    cpu_num = '0'
    channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
    dapi_channel = 3
    display_napari = 'True'
else:
    folder_pathway = sys.argv[1]
    threshold_folder = sys.argv[2]
    cpu_num = ''#sys.argv[3]
    channel_names = ast.literal_eval(sys.argv[4])
    dapi_channel = int(sys.argv[5])
    display_napari = sys.argv[6]

st_dev_multiplier = 2 #Could have it different for each channel, as with 2 it takes the top 2.5% of cells. Based on proportions of how many cells there are in the gastruloid.

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

total_cells = np.max(segmented_image) + 1 #Include background label for indexing (so label 1 at position 1)
#print("Total cells:", total_cells)


characterised_cells = {cell: {'pixel_count': 0, 'z_position': 0.0, 'fate': '', 'display_colour': [0, 0, 0], **{channel: 0.0 for channel in channel_names}} for cell in range(total_cells)}

def quantify_cell_fluorescence(image, segmented_image):
    #print(f"{datetime.now():%H:%M:%S} - Quantifying cell fluorescence...")

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
    #print(f"{datetime.now():%H:%M:%S} - Calculating z positions...")
    
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

def find_thresholds():
    #print(f"{datetime.now():%H:%M:%S} - Plotting channel intensities relative to DAPI...")

    def plot_3D(x_data, y_data, z_data, channel_name):

        # Create the figure and 3D axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Reshape data to fit the model
        X = np.column_stack((x_data, y_data))

        # Fit the model using Linear Regression
        model = LinearRegression(fit_intercept=False) #Forces passing through 0, 0, 0
        model.fit(X, z_data)

        # Extract coefficients and intercept
        a, b = model.coef_
        #c = model.intercept_

        # Create a grid for plotting the plane
        x_grid, y_grid = np.meshgrid(np.linspace(0, 250, 50), 
                                    np.linspace(0, y_max, 2))
        
        z_grid = a * x_grid + b * y_grid # + c

        # Calculate the predicted z values from the fitted plane
        z_pred = model.predict(X)

        # Calculate the residuals (differences between actual and predicted z values)
        residuals = z_data - z_pred

        # Calculate the standard deviation of the residuals
        std_dev = np.std(residuals)

        # Show the standard deviation on the plot (optional)
        #print(f"Standard deviation of z values from the plane: {std_dev:.2f}")

        c = st_dev_multiplier * std_dev

        threshold_plane = z_grid + c

        thresholds[channel_name] = [a, b, c]

        # Step 4: Plot the scatter plot
        ax.scatter3D(x_data, y_data, z_data, alpha=0.5, s=20)
        ax.plot_surface(x_grid, y_grid, threshold_plane, alpha=0.5, cmap='viridis')
        ax.set_xlabel('DAPI')
        ax.set_zlabel(channel_name)
        ax.set_ylabel('Z')
        ax.set_zlim(0, z_max)
    
    characterised_cells_list = list(characterised_cells.items())

    x_data = np.array([cell['DAPI'] for _, cell in characterised_cells_list])
    y_data = np.array([cell['z_position'] for _, cell in characterised_cells_list])

    y_max = total_z
    x_max = 250
    z_max = 100

    thresholds = {}

    for channel in range(total_channels):
        channel_name = channel_names[channel]
        if not channel_name == 'DAPI':
            z_data = np.array([cell[channel_name] for _, cell in characterised_cells_list])
            plot_3D(x_data, y_data, z_data, channel_name)
            #plt.show()

    return thresholds

def save_thresholds_to_csv():
    image_name = folder_pathway[len(threshold_folder):][1:]

    # File path for the CSV file
    csv_pathway = threshold_folder + "/thresholds.csv"

    # Check if the file exists to determine if headers need to be written
    try:
        with open(csv_pathway, mode='x', newline='') as file:
            # If the file doesn't exist, create it and add headers
            writer = csv.writer(file)
            writer.writerow(['image_name', 'channel', 'a', 'b', 'c'])
    except FileExistsError:
        pass

    # Append the data to the file
    with open(csv_pathway, mode='a', newline='') as file:
        writer = csv.writer(file)
        for channel, values in thresholds.items():
            a, b, c = values
            writer.writerow([image_name, channel, a, b, c])

z_slice_averages = find_average_z_slice_of_each_label(segmented_image)

normalised_channel_values = quantify_cell_fluorescence(image, segmented_image)

thresholds = find_thresholds()

#print(thresholds)
save_thresholds_to_csv()

def save_characterised_cells_to_csv(characterised_cells, folder_pathway):
    file_pathway = folder_pathway + '/characterised_cells' + cpu_num + '.csv'

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(characterised_cells, orient='index')

    # Ensure that the index is reset (optional, for better formatting)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cell_number'}, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_pathway, index=False)


save_characterised_cells_to_csv(characterised_cells, folder_pathway)

print("Thresholds succesfully saved")