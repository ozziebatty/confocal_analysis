import logging
import numpy as np
import tifffile
import napari
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]
dynamic_channel_thresholds = [1, -0.21, 0.13, -0.215]

files_to_analyse = {'/Users/oskar/Desktop/steventon_lab/image_analysis/images/very_cropped_gastruloid_z.tiff' : ('/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_very_cropped.tiff',  '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/very_cropped_gastruloid_z_characterised_cells.csv')}
                    

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Load images
logging.info("Loading images...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')
segmented_image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_semi_cropped.tiff')

# Obtain image properties
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
    'purple': (128, 0, 128)
}

def preprocess(image):
    logging.info("Preprocessing...")

    preprocessed_image = np.zeros_like(image)
    
    for z in range(total_z):
        z_slice = image[z]

        #Preprocess using adapthist
        for channel in range(total_channels):
            preprocessed_image[z][channel] = img_as_ubyte(equalize_adapthist(z_slice[channel], kernel_size=None, clip_limit=0.01, nbins=256))

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
            for i in range(len(channels)):
                characterised_cells[cell_number][f'channel_{i}'] = channels[i]

    for z_slice in z_slice_fluorescence:
        z_slice['channels'] /= z_slice['pixel_count']

    #Normalise cell intensities relative to DAPI
    logging.info("Normalising intensities to DAPI...")
    normalised_cell_fluorescence = np.zeros(len(cell_fluorescence), dtype=[('cell_number', int), ('channels', float, total_channels)])
    for cell in cell_fluorescence:
        
        cell_number = cell['cell_number']
        normal_value = cell['channels'][0]
        
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
    colours = ['red', 'green', 'blue']
    logging.info("Plotting channel intensities...")
    cell_fluorescence = cell_fluorescence[1:] #Cut background

    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']
    
    plt.figure(figsize=(10, 6))

    for i in range(1, 3):
        fluorescence = channel_fluorescence[:, i]
        plt.scatter(average_z, fluorescence, color=colours[i-1], s=10, label=f'Channel {i+1}')

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, i], 12)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z) + dynamic_channel_thresholds[i]
        plt.plot(average_z, best_fit_line, color='black')
        
        # Find outliers
        outliers = (fluorescence > best_fit_line)
        
        # Plot outliers
        plt.scatter(average_z[outliers], fluorescence[outliers], color='black', s=2, label=f'Above threshold Channel {i+1}', marker='x')

    # Customize the plot
    plt.xlabel('Z Slice')
    plt.ylabel('Normalised Fluorescence')
    plt.title('Z Slice vs. Normalized Fluorescence for Each Channel')
    plt.legend(title='Channels')
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

def assign_fates(cell_fluorescence, z_slice_average, characterised_cells):
    logging.info("Assigning cell fates...")    

    cell_fluorescence = cell_fluorescence[1:] #Cut background
    channel_fluorescence = cell_fluorescence['channels']
    average_z = z_slice_average['average_z']

    total_Bra, total_Sox2, total_Bra_Sox2, total_unlabelled = 0, 0, 0, 0

    for cell in range(1, total_cells):
        characterised_cells[cell]['fate'] = 'unlabelled' #Set to unlabelled by default
        total_unlabelled += 1
    
    for channel in range(1, 3):
        fluorescence = channel_fluorescence[:, channel]

        # Calculate the line of best fit
        coefficients = np.polyfit(average_z, channel_fluorescence[:, channel], 12)
        polynomial = np.poly1d(coefficients)
        best_fit_line = polynomial(average_z) + dynamic_channel_thresholds[channel]
        
        # Find outliers
        for cell in range(1, len(fluorescence)):
            if fluorescence[cell] > best_fit_line[cell]:
                if channel == 1:
                    characterised_cells[(cell+1)]['fate'] = 'Bra'
                    total_Bra += 1
                    total_unlabelled -= 1
                elif channel == 2:
                    if characterised_cells[(cell+1)]['fate'] == 'Bra':
                        characterised_cells[(cell+1)]['fate'] = 'Bra_Sox2'
                        total_Bra_Sox2 += 1
                        total_Bra -= 1                        
                    else:
                        characterised_cells[(cell+1)]['fate'] = 'Sox2'
                        total_Sox2 += 1
                        total_unlabelled -= 1

    print("Total Bra+ : ", total_Bra)
    print("Total Sox2+ : ", total_Sox2)
    print("Total Bra+ Sox2+ : ", total_Bra_Sox2)
    print("Total unlabelled : ", total_unlabelled)    

def create_characterised_image(characterised_cells):
    logging.info("Creating characterised image...")
    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)
    for cell, data, in characterised_cells.items():
        fate = data['fate']
        if fate == 'Bra_Sox2':
            colour = display_colours['purple']
        elif fate == 'Bra':
            colour = display_colours['red']
        elif fate == 'Sox2':
            colour = display_colours['green']
        else:
            colour = display_colours['blue']

        characterised_image[segmented_image == cell] = colour

    tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/characterised_gastruloid_z.tiff', characterised_image)

    #Visualise with Napari
    logging.info("Visualizing results with Napari...")
    viewer = napari.Viewer()
    viewer.add_image(characterised_image, name='Characterised image')
    viewer.add_labels(segmented_image, name='Segmentation Masks')
    napari.run()
    

    return characterised_image

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


#characterised_cells = characterise_cells(normalised_cell_fluorescence)
assign_fates(normalised_cell_fluorescence, z_slice_average, characterised_cells)

save_to_csv(characterised_cells, '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/semi_cropped_gastruloid_z_characterised_cells.csv')

#characterised_image = create_characterised_image(characterised_cells)


