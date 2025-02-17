from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - IMPORTING PACKAGES...")

import pandas as pd
import ast
import napari
import tifffile
import numpy as np


folder_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p1'
cpu_num = ''
segmented_image_pathway = folder_pathway + '/stitched.tiff'
segmented_image = tifffile.imread(segmented_image_pathway)
preprocessed_image_pathway = folder_pathway + '/preprocessed.tiff'
image = tifffile.imread(preprocessed_image_pathway)

total_cells = np.max(segmented_image) + 1 #Include a background for indexing (so label 1 at position 1)
print(total_cells)

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

def create_characterised_image(characterised_cells):
    print(f"{datetime.now():%H:%M:%S} - Creating characterised image...")

    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)

    #for cell, data, in characterised_cells.items():
        #characterised_image[segmented_image == cell] = data['display_colour']

    # Create a 2D array with the same shape as segmented_image that maps cell labels to color indices
    cell_indices = np.unique(segmented_image)

    # Create a lookup table for display colours
    colours = np.array([characterised_cells[cell]['display_colour'] for cell in np.unique(segmented_image)])

    # Use broadcasting to assign the colours directly
    # Use advanced indexing to directly assign colors based on the cell labels
    for i, cell in enumerate(np.unique(segmented_image)):
        characterised_image[segmented_image == cell] = colours[i]

    # Visualize with Napari
    print(f"{datetime.now():%H:%M:%S} - Visualizing results with Napari...")
    viewer = napari.Viewer()
    viewer.add_image(characterised_image)
    viewer.add_image(image)
    napari.run()

    return characterised_image

characterised_cells = load_characterised_cells_from_csv(folder_pathway)

characterised_image = create_characterised_image(characterised_cells)
