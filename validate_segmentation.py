import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import tifffile
import napari


cropping_degree = 'semi_cropped'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting image processing...")

#Load file
logging.info("Loading file pathways...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
segmented_image = tifffile.imread(file_pathways.loc[cropping_degree, 'segmented_stitched'])

flat_image = segmented_image.flatten()

# Get unique values and their frequency counts
logging.info("Counting pixels")
unique_numbers, counts = np.unique(flat_image, return_counts=True)

total_cells = len(np.unique(flat_image))

print(f"Total cells: {total_cells}")

# Skip the 0 value
non_zero_mask = unique_numbers != 0
unique_numbers_non_zero = unique_numbers[non_zero_mask]
counts_non_zero = counts[non_zero_mask]


mean_pixels = np.round(np.mean(counts_non_zero), 0)
mean_diameter = np.round(mean_pixels ** (1/3), 2)
std_dev_counts = np.round(np.std(counts_non_zero), 0)
variance = np.round(np.var(counts_non_zero), 0)

print(f"Mean pixels: {mean_pixels}")
print(f"Mean diameter: {mean_diameter}")
print(f"Standard Deviation: {std_dev_counts}")
print(f"Variance: {variance}")

# Count how many numbers have counts between 150 and 800
within_range_mask = (counts_non_zero >= mean_pixels / 3) & (counts_non_zero <= mean_pixels * 3)
count_within_range = np.sum(within_range_mask)

# Calculate the proportion
proportion_within_range = np.round(100*count_within_range / len(counts_non_zero), 1)

print(f"Reasonably sized cells: {proportion_within_range:.4f} %")

plt.scatter(unique_numbers_non_zero, counts_non_zero, s=2)
plt.xlabel('Cell')
plt.ylabel('Pixels')
plt.title('Frequency of Numbers in segmented_image')
plt.ylim(0, mean_pixels)
#plt.show()

logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_labels(segmented_image)
napari.run()

