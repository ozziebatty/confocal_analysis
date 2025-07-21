#import ast
print("Running")
from datetime import datetime
#import sys
import numpy as np
#from collections import Counter
#import csv
#import os
import napari
import tifffile
#import pandas as pd

debug_mode = True
condition = 'treatment'
#remove_channel = 'Sox2 Orange'
z_of_interest = 50

thresholds =  {
    'Sox1': [0.0876700063980764, 0.0481177417068815, 13.47],
               'Sox2 Cyan': [0.07612990877162337, 0.0250923265032827045, 6.60],
               'Sox2 Orange': [0.04518975473155211, 0.006305888409557325, 5.1160452844525395],
               'Bra': [0.03058175471629982, 0.017052924114643686, 10.14]}

fate_colours = {
        'bra+,': [255, 0, 0],  # Normalized key
        'unlabelled': [100, 100, 100],  # Normalized key
        'sox2 cyan+,': [50, 50, 255],  # Normalized key
        'sox1+,': [0, 255, 0],  # Normalized key
        'sox1+, sox2 cyan+,': [0, 200, 200],  # Normalized key
        'sox1+, bra+,': [255, 0, 255],  # Normalized key
        'sox2 cyan+, bra+': [255, 0, 255],  # Normalized key
        'sox1+, sox2 cyan+, bra+,': [255, 0, 255],  # Normalized key
        'background': [0, 0, 0]  # Normalized key
    }

characterised_cells_summary = {}

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p4'
    threshold_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results'
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

#segmented_image_pathway = folder_pathway + '/stitched.tiff'
#segmented_image = tifffile.imread(segmented_image_pathway)[z_of_interest]
preprocessed_image_pathway = folder_pathway + '/deconvoluted.tiff'
image = tifffile.imread(preprocessed_image_pathway)[z_of_interest]

def load_characterised_cells_from_csv(folder_pathway):
    file_pathway = folder_pathway + '/characterised_cells' + cpu_num + '.csv'
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(file_pathway)
    
    # Set the index to 'cell_number'
    df.set_index('cell_number', inplace=True)
    
    # Fill NaN values in the 'fate' column with an empty string
    if 'fate' in df.columns:
        df['fate'] = df['fate'].fillna('')

    # Normalize the 'fate' column for consistent matching
    df['fate_normalized'] = df['fate'].str.lower().str.strip()

    # Debug: Print the number of cells corresponding to each fate
    fate_counts = df['fate_normalized'].value_counts()
    #print("Number of cells corresponding to each fate:")
    #print(fate_counts)

    # Map the normalized 'fate' to the corresponding color
    df['display_colour'] = df['fate_normalized'].map(fate_colours)
    
    # Fill NaN values in 'display_colour' with the default 'Unlabelled' color
    df['display_colour'] = df['display_colour'].apply(lambda x: x if isinstance(x, list) else [100, 100, 100])

    # Convert the DataFrame back into the dictionary format
    characterised_cells = df.to_dict(orient='index')
    
    return characterised_cells

def assign_fates_by_intensity_to_threshold(characterised_cells):
    #print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    characterised_cells_list = list(characterised_cells.items())

    #Clear to initialise
    for cell in characterised_cells_list:
        DAPI_fluorescence = cell[1]['DAPI']
        z_position = cell[1]['z_position']

        bra, sox1, sox2 = False, False, False

        if cell[1]['Bra'] > thresholds['Bra'][0] * DAPI_fluorescence + thresholds['Bra'][1] * z_position + thresholds['Bra'][2]:
            bra = True
        if cell[1]['Sox1'] > thresholds['Sox1'][0] * DAPI_fluorescence + thresholds['Sox1'][1] * z_position + thresholds['Sox1'][2]:
            sox1 = True
        if cell[1]['Sox2 Cyan'] > thresholds['Sox2 Cyan'][0] * DAPI_fluorescence + thresholds['Sox2 Cyan'][1] * z_position + thresholds['Sox2 Cyan'][2]:
            sox2 = True

        if bra == True:
            if sox1 == True:
                if sox2 == True:
                    colour = [200, 100, 0] #Bra Sox1 Sox2
                    fate = 'NMP'
                else:
                    colour = [200, 100, 0] #Bra Sox1
                    fate =  'NMP'
            else:
                if sox2 == True:
                    colour = [200, 100, 0] #Bra Sox2
                    fate = 'NMP'
                else:
                    colour = [200, 0, 0] #Bra
                    fate = 'Mesoderm'
        elif sox1 == True:
            if sox2 == True:
                colour = [0, 150, 100] #Sox1 Sox2
                fate = 'Neural'
            else:
                colour = [0, 200, 0] #Sox1
                fate = 'Neural'
        elif sox2 == True:
            colour = [0, 100, 200] #Sox2
            fate = 'Plurpotent'
        else:
            colour = [70, 70, 70] #Unlabelled
            fate = 'Unlabelled'
                
        #print(characterised_cells[cell])
        #print(characterised_cells[cell[0]])
        #cell_label = characterised_cells[cell]

        #print(characterised_cells[cell[0]])


        characterised_cells[cell[0]]['display_colour'] = colour
        characterised_cells[cell[0]]['fate'] = fate

    """ 
    for channel in channel_names:
        if not channel == 'DAPI':
            if not channel == remove_channel:
                a, b, c = thresholds[channel]

                for cell in characterised_cells_list:
                    cell_label = cell[0]
                    DAPI_fluorescence = cell[1]['DAPI']
                    channel_fluorescence = cell[1][channel]

                    threshold = a * DAPI_fluorescence + b * z_position + c

                    if channel_fluorescence > threshold:
                        characterised_cells[cell_label]['fate'] += channel + '+, '
    """
    # Assign Background to label 0
    characterised_cells[0]['fate'] = 'background'
    characterised_cells[0]['display_colour'] = [0, 0, 0]

    """ 
    # If not assigned, assign 'Unlabelled'
    for cell_label in range(1, len(characterised_cells_list)):
        if characterised_cells[cell_label]['fate'] == '':  # Check if the fate string is empty
            characterised_cells[cell_label]['fate'] = 'unlabelled'

    # Assign display colours based on the fate_colours map
    for cell_label, cell_data in characterised_cells.items():
        fate = cell_data['fate'].strip()  # Remove any trailing whitespace
        if fate in fate_colours:
            cell_data['display_colour'] = fate_colours[fate]
        else:
            # Handle any unexpected fates (optional)
            cell_data['display_colour'] = fate_colours['unlabelled']
    """
    # Count fates
    fates = [cell_data['fate'].strip() for cell_data in characterised_cells.values()]
    fate_counts = Counter(fates)

    for fate, count in fate_counts.items():
        if not fate == 'background':
            characterised_cells_summary[fate] = count

    print(characterised_cells_summary)
    return characterised_cells

def save_characterised_cells_summary_to_csv(characterised_cells_summary, folder_pathway):
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

    characterised_image_pathway = folder_pathway + '/characterised_image.tiff'

    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)

    # Create a 2D array with the same shape as segmented_image that maps cell labels to color indices
    cell_indices = np.unique(segmented_image)

    # Create a lookup table for display colours
    colours = np.array([characterised_cells.get(cell, {}).get('display_colour', [70, 70, 70]) for cell in cell_indices])

    # Use broadcasting to assign the colours directly
    for i, cell in enumerate(cell_indices):
        mask = segmented_image == cell
        if mask.any():  # Check if the cell exists in the segmented image
            characterised_image[mask] = colours[i]

    # Visualize with Napari
    print(f"{datetime.now():%H:%M:%S} - Visualizing results with Napari...")
    #viewer = napari.Viewer()
    #viewer.add_image(characterised_image)
    #viewer.add_image(image)
    #napari.run()

    return characterised_image

def multichannel_display(image):
    print(f"{datetime.now():%H:%M:%S} - Loading into Napari...")

    colours = ['green', 'cyan', 'orange', 'grey', 'red']
    viewer = napari.Viewer()

    print(image.shape)

    #viewer.add_image(characterised_image)

    # Add each channel as a separate layer with its own colour
    for channel, colour in enumerate(colours):
        viewer.add_image(
            image[channel, :, :], 
            name=channel_names[channel],
            colormap=colour,
            blending='additive',  # Use additive blending for transparency
            opacity=0.7  # Adjust opacity as needed
        )

    napari.run()


#characterised_cells = load_characterised_cells_from_csv(folder_pathway)

#characterised_cells = assign_fates_by_intensity_to_threshold(characterised_cells)

#characterised_image = create_characterised_image(characterised_cells)
multichannel_display(image)


#print(characterised_cells)

#characterised_cells_summary = assign_fates_by_intensity_to_threshold(characterised_cells)

#save_characterised_cells_summary_to_csv(characterised_cells_summary, folder_pathway)