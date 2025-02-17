import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import tifffile


cropping_degree = 'semi_cropped'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting image processing...")

#Load file
logging.info("Loading file pathways...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
segmented_image = tifffile.imread(file_pathways.loc[cropping_degree, 'segmented_tiff'])

unique_cells = np.unique(segmented_image)
unique_cells = unique_cells[unique_cells != 0]

total_cells = len(unique_cells)

total_z = segmented_image.shape[0]

for cell in unique_cells:
    x_values = []
    y_values = []
    for z in range(total_z):
        if np.any(segmented_image[z] == cell):
            first_z = z
            break
    
    # Count instances of the label in each z slice starting from the first appearance
    for z in range(first_z, total_z):
        count = np.sum(segmented_image[z] == cell)
        x_values.append(z - first_z + 1)  # relative z slice index
        y_values.append(count)

    # Plot the scatter plot
    plt.clf()
    plt.plot(x_values, y_values)
    plt.xlabel('Relative Z Slice (starting from first appearance)')
    plt.ylabel('Number of Instances')
    plt.title('Scatter Plot of Cell Counts Across Z Slices')
    plt.xlim(0,10)
    plt.show()

print(f"Total cells: {total_cells}")

# Plot the scatter plot
plt.scatter(x_values, y_values, s=2)
plt.xlabel('Relative Z Slice (starting from first appearance)')
plt.ylabel('Number of Instances')
plt.title('Scatter Plot of Cell Counts Across Z Slices')
plt.show()
