import pandas as pd
import numpy as np
from skimage import io, filters, measure, morphology, exposure
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import tifffile

# Read an image from file
image = io.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/gastruloid_z.lsm')

# Define the ROI (Region of Interest)
z_start, z_end = 10, 170  # Example slices
y_start, y_end = 40, 490
x_start, x_end = 40, 460

# Crop the image to the ROI
roi_image = image[z_start:z_end, x_start:x_end, y_start:y_end]

# Normalize the image
roi_image = (roi_image - np.min(roi_image)) / (np.max(roi_image) - np.min(roi_image))

# Apply Gaussian filtering to reduce noise
filtered_roi_image = gaussian(roi_image, sigma=1, mode='constant')

# Enhance contrast using histogram equalization
enhanced_image = exposure.equalize_adapthist(filtered_roi_image)

# Convert enhanced image to binary mask using Otsu's method
binary_mask = enhanced_image[..., 0] > filters.threshold_otsu(enhanced_image[..., 0])

# Compute distance transform
distance = distance_transform_edt(binary_mask)

# Find markers as local maxima in the distance transform
local_maxi = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=binary_mask)
markers = np.zeros_like(distance, dtype=int)
markers[tuple(local_maxi.T)] = np.arange(1, local_maxi.shape[0] + 1)

# Apply watershed segmentation
labels = watershed(-distance, markers, mask=binary_mask)

# Remove small objects
cleaned_labels = remove_small_objects(labels, min_size=100)

# Ensure cleaned_labels is of integer type for regionprops
cleaned_labels = cleaned_labels.astype(np.int32)

# Measure properties of labeled image regions for each channel
average_fluorescence = []
for channel_index in range(roi_image.shape[-1]):
    properties = measure.regionprops(cleaned_labels, intensity_image=roi_image[..., channel_index])

    # Calculate average fluorescence for each cell in the current channel
    channel_fluorescence = []
    for prop in properties:
        cell_label = prop.label
        mean_intensity = prop.mean_intensity
        channel_fluorescence.append((cell_label, channel_index, mean_intensity))

    average_fluorescence.extend(channel_fluorescence)

# Convert the list to a DataFrame
average_fluorescence_df = pd.DataFrame(average_fluorescence, columns=['Cell Label', 'Channel', 'Average Fluorescence'])

# Initialize a color map for each cell
color_map = {label: (0, 0, 0) for label in np.unique(cleaned_labels)}  # Initialize all as black

# Iterate over each cell to determine the color
for cell_label in np.unique(cleaned_labels):
    # Extract average fluorescence values for each channel for this cell
    ch0_values = average_fluorescence_df.query(f'`Cell Label` == {cell_label} and `Channel` == 0')['Average Fluorescence']
    ch1_values = average_fluorescence_df.query(f'`Cell Label` == {cell_label} and `Channel` == 1')['Average Fluorescence']
    ch2_values = average_fluorescence_df.query(f'`Cell Label` == {cell_label} and `Channel` == 2')['Average Fluorescence']

    # Default to 0 if not found
    ch0_fluorescence = ch0_values.values[0] if not ch0_values.empty else 0
    ch1_fluorescence = ch1_values.values[0] if not ch1_values.empty else 0
    ch2_fluorescence = ch2_values.values[0] if not ch2_values.empty else 0

    #print(f"Label {cell_label} - ch0: {ch0_fluorescence}, ch1: {ch1_fluorescence}, ch2: {ch2_fluorescence}")



    # Assign colors based on conditions
    if ch0_fluorescence < 0.05:
        color_map[cell_label] = (0, 0, 0)  # Black
    elif ch1_fluorescence > ch2_fluorescence:
        color_map[cell_label] = (1, 0, 0)  # Red
    elif ch2_fluorescence > ch1_fluorescence:
        color_map[cell_label] = (0, 1, 0)  # Green
    else:
        color_map[cell_label] = (0, 0, 1)  # Blue

    #print(f"Color assigned to label {cell_label}: {color_map[cell_label]}")


# Create a new image stack with colored cells
colored_stack = np.zeros((*cleaned_labels.shape, 3))  # RGB image stack

for z in range(colored_stack.shape[0]):
    for y in range(colored_stack.shape[1]):
        for x in range(colored_stack.shape[2]):
            label = cleaned_labels[z, y, x]
            if label > 0:
                colored_stack[z, y, x] = color_map[label]

# Save the colored z-stack image
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/colored_gastruloid_z.tiff', colored_stack)

# Visualize the segmentation
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(binary_mask[0], cmap='gray')
ax[0].set_title('Binary Mask (First Slice)')

ax[1].imshow(distance[0], cmap='jet')
ax[1].set_title('Distance Transform (First Slice)')

ax[2].imshow(colored_stack[0])
ax[2].set_title('Colored Segmentation (First Slice)')
plt.show()
