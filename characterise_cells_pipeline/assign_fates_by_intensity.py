import ast
import sys
import numpy as np
from collections import Counter
import csv
import os
import pandas as pd

debug_mode = False
condition = 'control'
remove_channel = 'Sox2 Orange'

thresholds =  {
    'Sox1': [0.12976700063980764, 0.07381177417068815, 27.47],
               'Sox2 Cyan': [0.22612990877162337, 0.10923265032827045, 28.60],
               'Sox2 Orange': [0.04518975473155211, 0.006305888409557325, 5.1160452844525395],
               'Bra': [0.0808175471629982, 0.054052924114643686, 18.14]}

channel_colours = ['green', 'blue', 'orange', 'grey', 'red']

display_colours = {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'grey': [100, 100, 100],
    'purple': [150, 0, 150],
    'orange': [150, 150, 0]
}

characterised_cells_summary = {}

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p2'
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

def load_characterised_cells_from_csv(folder_pathway):
    file_pathway = folder_pathway + '/characterised_cells' + cpu_num + '.csv'
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(file_pathway)
    
    # Set the index to 'cell_number'
    df.set_index('cell_number', inplace=True)
    
    # Fill NaN values in the 'fate' column with an empty string
    if 'fate' in df.columns:
        df['fate'] = df['fate'].fillna('')


    # Convert 'display_colour' from string back to a list
    if 'display_colour' in df.columns:
        df['display_colour'] = df['display_colour'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    

    # Convert the DataFrame back into the dictionary format
    characterised_cells = df.to_dict(orient='index')
    
    return characterised_cells

def assign_fates_by_intensity_to_threshold(characterised_cells):
    #print(f"{datetime.now():%H:%M:%S} - Assigning cell fates...")    

    characterised_cells_list = list(characterised_cells.items())

    for channel in channel_names:
        #print(channel)
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
                        #print(characterised_cells[cell_label]['fate'])
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
    return characterised_cells_summary

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

characterised_cells = load_characterised_cells_from_csv(folder_pathway)

#print(characterised_cells)

characterised_cells_summary = assign_fates_by_intensity_to_threshold(characterised_cells)

save_characterised_cells_summary_to_csv(characterised_cells_summary, folder_pathway)