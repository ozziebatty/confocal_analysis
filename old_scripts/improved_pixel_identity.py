import numpy as np
import tifffile
import pandas as pd
from skimage.filters import gaussian, threshold_otsu
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')

# Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]

# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

def categorize_pixels(channel_images, x_pixels, y_pixels):
    categorized_image = np.zeros((x_pixels, y_pixels, 3), dtype=np.uint8)  # 3 channels for RGB
    category_counts = {'Black': [], 'Purple': [], 'Red': [], 'Green': [], 'Blue': []}

    for x in range(x_pixels):
        for y in range(y_pixels):
            # Get the pixel values for the three channels
            channel_0_value = channel_images[0][x, y]  # DAPI
            channel_1_value = channel_images[1][x, y]  # Bra
            channel_2_value = channel_images[2][x, y]  # Sox2
            relative_channel_1 = channel_1_value / (channel_0_value + 0.01)
            relative_channel_2 = channel_2_value / (channel_0_value + 0.01)
            black_threshold = 50
            red_threshold = 0.5
            green_threshold = 0.45

            if channel_0_value < black_threshold:
                # Black for pixels where channel 0 value is less than the threshold
                categorized_image[x, y] = (0, 0, 0)
                category_counts['Black'].append([channel_0_value, channel_1_value, channel_2_value])
            elif relative_channel_1 > red_threshold:
                # Red for pixels where channel 1 is higher than threshold
                if relative_channel_2 > green_threshold:
                    categorized_image[x, y] = (128, 0, 128)  # NMP (overlaps)
                    category_counts['Purple'].append([channel_0_value, channel_1_value, channel_2_value])
                else:
                    categorized_image[x, y] = (255, 0, 0)
                    category_counts['Red'].append([channel_0_value, channel_1_value, channel_2_value])
            elif relative_channel_2 > green_threshold:
                # Green for pixels where channel 2 is higher than threshold
                categorized_image[x, y] = (0, 255, 0)
                category_counts['Green'].append([channel_0_value, channel_1_value, channel_2_value])
            else:
                # Blue for remaining pixels
                categorized_image[x, y] = (0, 0, 255)
                category_counts['Blue'].append([channel_0_value, channel_1_value, channel_2_value])

    return categorized_image, category_counts

# Dictionary to accumulate pixel values per category across all z-slices
accumulated_counts = {'Black': [], 'Purple': [], 'Red': [], 'Green': [], 'Blue': []}

# Create empty stacks for the categorized images
categorized_stack = np.zeros((total_z, x_pixels, y_pixels, 3), dtype=np.uint8)
filtered_image = np.zeros_like(image)
equalized_image = np.zeros_like(image)

# Process each z-slice
for z in range(total_z):
    for ch in range(total_channels):
        # Apply adaptive histogram equalization
        equalized_slice = equalize_adapthist(image[z, ch], kernel_size=None, clip_limit=0.01, nbins=256)
        equalized_image[z] = img_as_ubyte(equalized_slice)

        # Apply Gaussian filter
        filtered_image[z, ch] = gaussian(equalized_image[z, ch], sigma=1, mode='constant', preserve_range=True)

    # Threshold and create binary mask for the first channel
    binary_mask = filtered_image[z, 0] > threshold_otsu(filtered_image[z, 0])

    # Apply binary mask to channels for categorization
    channel_images = [filtered_image[z, ch] * binary_mask for ch in range(total_channels)]
    categorized_image, category_counts = categorize_pixels(channel_images, x_pixels, y_pixels)

    # Add the categorized image to the stack
    categorized_stack[z] = categorized_image

    # Accumulate counts for the DataFrame
    for category in accumulated_counts:
        accumulated_counts[category].extend(category_counts[category])


# Calculate average pixel values for each category
average_pixel_values = {}
for category in accumulated_counts:
    values = np.array(accumulated_counts[category])
    if len(values) > 0:
        average_pixel_values[category] = values.mean(axis=0)
    else:
        average_pixel_values[category] = np.array([0, 0, 0])  # In case there are no pixels for a category

# Create a DataFrame
df = pd.DataFrame(average_pixel_values, index=channel_names[:3]).T
print("\nAverage Pixel Values for Each Category:")
print(df)

# Save the entire categorized stack as a single multi-channel TIFF
output_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/categorized_stack_pixels_semi_cropped.tiff'
tifffile.imwrite(output_path, categorized_stack, photometric='rgb')

print("\nComplete")
