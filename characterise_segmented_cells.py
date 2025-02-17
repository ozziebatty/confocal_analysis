import logging
import numpy as np
import tifffile
import napari
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import csv
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform

cropping_degree = 'semi_cropped'

# Define channel names
channel_names = ["Sox1", "Bra", "DAPI", "OPP"]
dynamic_channel_thresholds = [0.15, 0.05, 0, -0.215] #Sox2 was 0.13
channel_stdev_thresholds = [-7, -2, 1, -1] #from 1, -6, 3, -1, more negative is more generous

dapi_channel = 2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Load images
logging.info("Loading images...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
segmented_image = tifffile.imread(file_pathways.loc[cropping_degree, 'segmented_stitched'])
image = tifffile.imread(file_pathways.loc[cropping_degree, 'resized'])


normalised_cell_fluorescence = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/normalised_cell_fluorescence_OPP_p5_tall.csv')

z_slice_fluorescence = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/z_slice_averages_OPP_p5_tall.csv')
formatted_z_slice_fluorescence = np.array([f"({repr(tup[0])}, {tup[1]})" for tup in z_slice_fluorescence])
z_slice_fluorescence = formatted_z_slice_fluorescence

#print(normalised_cell_fluorescence)
#print(z_slice_fluorescence)

#Image properties prior to resizing
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

print(total_channels)
print(total_z)

#Resize image
##scale_factor = 4
##resized_image = np.zeros((total_z, total_channels, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.float32)
##
##for z in range(total_z):
##    resized_z = np.zeros((total_channels, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.float32) 
##    for channel in range(total_channels):
##        resized_z_channel = skimage.transform.resize(image[z][channel], (y_pixels * scale_factor, x_pixels * scale_factor), anti_aliasing=True)
##        resized_z[channel] = resized_z_channel
##    resized_image[z] = resized_z
##
##image = resized_image

# Image properties post resizing
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]
total_cells =len(np.unique(segmented_image))

print(f"Image shape: {image.shape} in ZCYX")
print(f"Segmented image shape: {segmented_image.shape} in ZYX")
print(total_cells - 1, " cells to characterise") #Exclude the background

#Initialise characterised_cells
characterised_cells = {cell: {'pixel_count': 0, 'fate': 'Unlabelled'} for cell in range(1, total_cells + 1)}

# Define representative colours for each range
display_colours = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'purple': (128, 0, 128),
    'grey': (100, 100, 100)
}

def preprocess(image):
    logging.info("Preprocessing...")

    preprocessed_image = np.zeros_like(image)
    
    for z in range(total_z):
        z_slice = image[z]

        z_slice_normalized = (z_slice - z_slice.min()) / (z_slice.max() - z_slice.min())

        #Preprocess using adapthist
        for channel in range(total_channels):
            preprocessed_image[z][channel] = img_as_ubyte(equalize_adapthist(z_slice_normalized[channel], kernel_size=31, clip_limit=0.0021, nbins=256))

    return preprocessed_image


def quantify_cell_fluorescence(image, segmented_image):
    logging.info("Quantifying cell fluorescence...")

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
    logging.info("Normalising intensities to DAPI...")
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

def find_average_z_slice_of_each_label(segmented_image):
    logging.info("Calculating z positions...")
    z_slice_averages = np.zeros(total_cells, dtype=[('running_z_total', int), ('z_stack_count', int), ('average_z', float)])

    for z in range(total_z):
        z_slice_segmented_image = segmented_image[z]
        
        for label in np.unique(z_slice_segmented_image):
            if label == 0:  # Skip background
                continue
            z_slice_averages[label]['running_z_total'] += z
            z_slice_averages[label]['z_stack_count'] += 1

    z_slice_averages = z_slice_averages[1:]

    for label in range(len(z_slice_averages)):
        z_slice_averages[label]['average_z'] = z_slice_averages[label]['running_z_total'] / z_slice_averages[label]['z_stack_count']
      
    return z_slice_averages

def plot_channel_intensities_to_z(cell_fluorescence, z_slice_average):
    colours = ['red', 'green', 'blue', 'orange']
    logging.info("Plotting channel intensities...")
    cell_fluorescence = cell_fluorescence[1:] #Cut background

    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']
    
    plt.figure(figsize=(10, 6))

    for channel in range(0, 1):
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
    plt.show()

def assign_fates_by_threshold(cell_fluorescence, z_slice_average, characterised_cells):
    logging.info("Assigning cell fates...")    

    cell_fluorescence = cell_fluorescence[1:] #Cut background
    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    total_Bra, total_Sox1, total_Bra_Sox1, total_unlabelled = 0, 0, 0, 0

    for cell in range(1, total_cells):
        characterised_cells[cell]['fate'] = 'unlabelled' #Set to unlabelled by default
        total_unlabelled += 1
    
    for channel in range(0, 3):
        fluorescence = channel_fluorescence[:, channel]

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 4)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z)

        # Calculate the standard deviation of the best fit line
        std_best_fit = np.std(best_fit_line)
        threshold = best_fit_line + dynamic_channel_thresholds[channel]

        
        # Find outliers
        for cell in range(1, len(fluorescence)):
            if fluorescence[cell] > threshold[cell]:
                if channel == 1:
                    characterised_cells[(cell+1)]['fate'] = 'Bra'
                    total_Bra += 1
                    total_unlabelled -= 1
                elif channel == 2:
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

def assign_fates_by_relative_intensity(cell_fluorescence, z_slice_average, characterised_cells):
    logging.info("Assigning cell fates...")    

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
    logging.info("Creating characterised image...")

    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)


    for cell, data, in characterised_cells.items():
        fate = data['fate']
        if fate == 'Bra_Sox1':
            colour = display_colours['purple']
        elif fate == 'Bra':
            colour = display_colours['red']
        elif fate == 'Sox1':
            colour = display_colours['green']
        else:
            colour = display_colours['grey']

        characterised_image[segmented_image == cell] = colour


    # Visualize with Napari
    logging.info("Visualizing results with Napari...")
    viewer = napari.Viewer()
    viewer.add_image(characterised_image)
    viewer.add_image(segmented_image, name='Segmentation Masks')
    napari.run()

    return characterised_image

    



    #return characterised_image

def save_to_csv(characterised_cells, file_path):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(characterised_cells, orient='index')

    # Ensure that the index is reset (optional, for better formatting)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cell_number'}, inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

z_slice_average = find_average_z_slice_of_each_label(segmented_image)


preprocessed_image = preprocess(image)
normalised_cell_fluorescence, z_slice_fluorescence = quantify_cell_fluorescence(preprocessed_image, segmented_image)

plot_channel_intensities_to_z(normalised_cell_fluorescence, z_slice_average)

assign_fates_by_threshold(normalised_cell_fluorescence, z_slice_average, characterised_cells)

#assign_fates_by_relative_intensity(normalised_cell_fluorescence, z_slice_average, characterised_cells)

save_to_csv(characterised_cells, '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/gastruloid_z_characterised_cells.csv')

plt.show()

characterised_image = create_characterised_image(characterised_cells)


