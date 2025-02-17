import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
import pandas as pd
from scipy.stats import pearsonr


# Read an image from file
gastruloid = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/gastruloid_z.lsm')

image = gastruloid

#Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]

# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
x_pixels = image.shape[2]
y_pixels = image.shape[3]


# Upper triangular matrix (including diagonal) of size 4x4 has 10 elements
pc_values = np.zeros((total_channels, total_channels, total_z))
mo_values = np.zeros((total_channels, total_channels, total_z))


# Extract z slice and channel from raw lsm
def image_snap(image, z_slice, channel):
    working_image = image[z_slice, channel, :, :]
    working_image_normalized = cv2.normalize(working_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return working_image_normalized

def create_mask(image, z_slice):
    #Create a mask for a single channel
    def create_mask_single_channel(image):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Apply Otsu's thresholding
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_mask

    # Create a composite mask from all channels
    def create_composite_mask(image, z_slice):
        composite_mask = np.zeros((x_pixels, y_pixels), dtype=np.uint8)
        for channel in range(total_channels):
            channel_image = image_snap(image, z_slice, channel)
            channel_mask = create_mask_single_channel(channel_image)
            composite_mask = cv2.bitwise_or(composite_mask, channel_mask)
        return composite_mask

    mask = create_composite_mask(image, z_slice)

    return mask

def apply_mask(image, z_slice, channel):
    image_slice = image_snap(image, z_slice, channel)
    masked_image = cv2.bitwise_and(image_slice, image_slice, mask=create_mask(image, z_slice))
    return masked_image

#Apply mask to each channel
def mask_each_channel(image, z_slice):
    masked_channel_images = []
    for channel in range(total_channels):
        masked_channel_image = apply_mask(image, z_slice, channel)
        masked_channel_images.append(masked_channel_image)
    return masked_channel_images

# Compute Pearson correlation coefficient for colocalization
def compute_pearson_correlation(channelA, channelB):
    flat_channelA = channelA.flatten()
    flat_channelB = channelB.flatten()
    correlation, _ = pearsonr(flat_channelA, flat_channelB)
    return correlation

# Compute Manders' overlap coefficient for colocalization
def compute_manders_overlap(channelA, channelB):
    intersection = np.sum((channelA > 0) & (channelB > 0))
    min_sum = np.sum(channelA > 0) + np.sum(channelB > 0)
    overlap_coefficient = intersection / min_sum
    return overlap_coefficient

# Display image
def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Calculate and print colocalization metrics for each pair of channels

def calculate_colocalisation(channel_images):    
    for i in range(len(channel_images)):
        for j in range(i+1, len(channel_images)):
            pearson_corr = compute_pearson_correlation(channel_images[i], channel_images[j])
            manders_overlap = compute_manders_overlap(channel_images[i], channel_images[j])

            pc_values[i, j, z_slice] = pearson_corr
            pc_values[j, i, z_slice] = pearson_corr  # Symmetric matrix
            mo_values[i, j, z_slice] = manders_overlap
            mo_values[j, i, z_slice] = manders_overlap  # Symmetric matrix

for z_slice in range(total_z):
    masked_channel_images = mask_each_channel(image, z_slice)
    calculate_colocalisation(masked_channel_images)


def flatten_matrix(matrix, total_channels, total_z, channel_names, correlation_type):
    total_channels, _, total_z = matrix.shape
    data= []

    for z in range(total_z):
        for i in range(total_channels):
            for j in range(i + 1, total_channels): # Only need the upper triangle and diagonal
                data.append({
                    'Z-slice': z,
                    'Channel 1': channel_names[i],
                    'Channel 2': channel_names[j],
                    f'{correlation_type} Value': matrix[i, j, z]
                })

    df = pd.DataFrame(data)
    return df


pc_df = flatten_matrix(pc_values, total_channels, total_z, channel_names, 'Pearson Correlation')
mo_df = flatten_matrix(mo_values, total_channels, total_z, channel_names, 'Manders Overlap')
merged_df = pd.merge(pc_df, mo_df, on=['Z-slice', 'Channel 1', 'Channel 2'])
merged_df.to_csv("/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/correlations.csv", index=False)


print("\n")
print("Complete")
