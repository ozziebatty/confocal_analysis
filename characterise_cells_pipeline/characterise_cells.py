from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - Importing packages for characterise_cells...")

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
    file_name = sys.argv[2]
    cpu_num = ''#sys.argv[3]
    channel_names = sys.argv[3]
    dapi_channel = int(sys.argv[4])
    display_napari = sys.argv[5]

st_dev_multiplier = 1.85 #Could have it different for each channel, as with 2 it takes the top 2.5% of cells. Based on proportions of how many cells there are in the gastruloid.

channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
channel_colours = ['green', 'blue', 'orange', 'grey', 'red']
display_colours = {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'grey': [100, 100, 100],
    'purple': [150, 0, 150]
}

print(f"{datetime.now():%H:%M:%S} - Loading images...")
image_pathway = folder_pathway + '/preprocessed' + cpu_num + '.tiff'
segmented_image_pathway = folder_pathway + '/stitched' + cpu_num + '.tiff'
image = tifffile.imread(image_pathway)
segmented_image = tifffile.imread(segmented_image_pathway)

print(f"{datetime.now():%H:%M:%S} - Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)
print(f"{datetime.now():%H:%M:%S} - Segmented image shape (z, c, y, x) :", segmented_image.shape, "dtype:", segmented_image.dtype)


total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

def clean_labels(segmented_image):
    # This function relabels the segmented image so that no labels are skipped.
    print(f"{datetime.now():%H:%M:%S} - Cleaning labels...")

    # Get the unique values in the segmented image, excluding 0 (background, if applicable)
    unique_labels = np.unique(segmented_image)
    
    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Relabel the segmented image using the mapping
    relabeled_image = np.vectorize(label_mapping.get)(segmented_image)
    
    return relabeled_image
    #Relabels segmented image so that every cell has a unique label and none are skipped

segmented_image = clean_labels(segmented_image)

total_cells = np.max(segmented_image) + 1 #Include a background for indexing (so label 1 at position 1)
print(total_cells , " cells to characterise")

characterised_cells = {cell: {'pixel_count': 0, 'z_position': 0.0, 'fate': '', 'display_colour': [0, 0, 0], **{channel: 0.0 for channel in channel_names}} for cell in range(total_cells)}

def quantify_cell_fluorescence(image, segmented_image):
    print(f"{datetime.now():%H:%M:%S} - Quantifying cell fluorescence...")

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
    print(f"{datetime.now():%H:%M:%S} - Normalising intensities to DAPI...")
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

    return normalised_channel_values, z_slice_fluorescence

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

def plot_channel_intensities_to_z(cell_fluorescence, z_slice_averages):
    print(f"{datetime.now():%H:%M:%S} - Plotting channel intensities...")

    colours = ['red', 'green', 'blue', 'orange']
    cell_fluorescence = cell_fluorescence[1:] #Cut background

    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_averages['average_z']
    
    plt.figure(figsize=(10, 6))

    for channel in range(total_channels):
        fluorescence = channel_fluorescence[:, channel]
        plt.scatter(average_z, fluorescence, color=colours[channel-1], s=10, label=f'Channel {channel+1}')

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + std_best_fit * dynamic_channel_thresholds[channel]

        # Plot the threshold
        plt.plot(average_z, threshold, label=f'Threshold - Channel {channel}')
        
        # Find outliers
        outliers = (fluorescence > threshold)
        
        # Plot outliers
        plt.scatter(average_z[outliers], fluorescence[outliers], color='black', s=2, marker='x')

    # Customize the plot
    plt.xlabel('Z Slice')
    plt.ylabel('Normalised Channel Intensity')
    plt.legend(title='Channels')
    plt.grid(True)

    # Display the plot
    plt.tight_layout()

def find_thresholds():
    print(f"{datetime.now():%H:%M:%S} - Plotting channel intensities relative to DAPI...")

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

        #print("a", a)
        #print("b", b)
        #print("c", c)


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

def assign_fates_by_relative_intensity_to_threshold(cell_fluorescence, z_slice_averages, characterised_cells):
    print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    #DAPI_values = channel_fluorescence[:, ]

    print(cell_fluorescence)

    for channel in range(total_channels):
        channel_name = channel_names[channel]
        if not channel_name == 'DAPI':
            fluorescence = channel_fluorescence[:, channel]

            a, b, c = threshold[channel_name]
            
            threshold = channel_thresholds[channel]
            #print(threshold)

            #Assign fate
            for cell in range(1, len(fluorescence)):
                if fluorescence[cell] > threshold:
                    characterised_cells[(cell)]['fate'] += channel_name + '+ '
                    original_colour = characterised_cells[(cell)]['display_colour']
                    new_colour = display_colours[channel_colours[channel]]
                    blended_colour = np.add(original_colour, new_colour).tolist()
                    characterised_cells[(cell)]['display_colour'] = blended_colour


    #If not assigned, assign 'Unlabelled'
    for cell in range(1, len(fluorescence)):
        if characterised_cells[cell]['fate'] == '':  # Check if the fate string is empty
            characterised_cells[cell]['fate'] = 'Unlabelled'
            characterised_cells[cell]['display_colour'] = display_colours['grey']

    #Count fates
    # Extract all fates into a list
    fates = [cell_data['fate'] for cell_data in characterised_cells.values()]
    fate_counts = Counter(fates)

    for fate, count in fate_counts.items():
        print(f"{fate}: {count}")

    print(pd.DataFrame(characterised_cells))

def assign_fates_by_intensity_to_threshold(characterised_cells):
    print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    characterised_cells_list = list(characterised_cells.items())

    for channel in channel_names:
        if not channel == 'DAPI':
            a, b, c = thresholds[channel]

            for cell in characterised_cells_list:
                cell_label = cell[0]
                DAPI_fluorescence = cell[1]['DAPI']
                channel_fluorescence = cell[1][channel]
                z_position = cell[1]['z_position']

                threshold = a * DAPI_fluorescence + b * z_position + c

                if channel_fluorescence > threshold:
                    characterised_cells[cell_label]['fate'] += channel + '+ '
                    original_colour = characterised_cells[cell_label]['display_colour']
                    new_colour = display_colours[channel_colours[channel_names.index(channel)]]
                    blended_colour = np.add(original_colour, new_colour).tolist()
                    characterised_cells[cell_label]['display_colour'] = blended_colour

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
        print(f"{fate}: {count}")

def assign_fates_by_relative_intensity(cell_fluorescence, z_slice_averages, characterised_cells):
    print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    cell_fluorescence = cell_fluorescence[1:] #Cut background
    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    total_Bra, total_Sox1, total_Bra_Sox1, total_unlabelled = 0, 0, 0, 0

    for cell in range(1, total_cells):
        characterised_cells[cell]['fate'] = 'unlabelled' #Set to unlabelled by default
        total_unlabelled += 1

    channel_1_over_2 = channel_fluorescence[:, 1] / channel_fluorescence[:, 2]

    plt.scatter(average_z, channel_1_over_2, s=2, label=f'Channel 1 relative to Channel 2')
    
    
    for channel in range(1, 3):
        fluorescence = channel_fluorescence[:, channel]

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + std_best_fit * dynamic_channel_thresholds[channel]

        
        # Find outliers
        for cell in range(1, len(fluorescence)):
            if fluorescence[cell] > threshold[cell]:
                if channel == 1:
                    characterised_cells[(cell+1)]['fate'] = 'Bra'
                    total_Bra += 1
                    total_unlabelled -= 1
                elif channel == 0:
                    if characterised_cells[(cell+1)]['fate'] == 'Bra':
                        characterised_cells[(cell+1)]['fate'] = 'Bra_Sox1'
                        total_Bra_Sox1 += 1
                        total_Bra -= 1                        
                    else:
                        characterised_cells[(cell+1)]['fate'] = 'Sox1'
                        total_Sox1 += 1
                        total_unlabelled -= 1

    print("Total Bra+ : ", total_Bra)
    print("Total Sox1+ : ", total_Sox1)
    print("Total Bra+ Sox1+ : ", total_Bra_Sox1)
    print("Total unlabelled : ", total_unlabelled) 

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

def save_characterised_cells_to_csv(characterised_cells, file_path):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(characterised_cells, orient='index')

    # Ensure that the index is reset (optional, for better formatting)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cell_number'}, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

def save_thresholds_to_csv():

# Image name for this data
    image_name = file_name

    # File path for the CSV file
    csv_pathway = folder_pathway + "/thresholds.csv"

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

normalised_channel_values, z_slice_fluorescence = quantify_cell_fluorescence(image, segmented_image)

thresholds = find_thresholds()

print(thresholds)

save_thresholds_to_csv()

#assign_fates_by_intensity_to_threshold(characterised_cells)

#characterised_image = create_characterised_image(characterised_cells)


print("Cell characterisation succesfully complete")