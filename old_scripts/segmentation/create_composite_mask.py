import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
import pandas as pd
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read an image from file
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')

print(f"Image shape: {image.shape}")

image = image[:, 0, :, :]


# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
x_pixels = image.shape[2]
y_pixels = image.shape[3]
z_slice = 60

masked = np.zeros((total_z, x_pixels, y_pixels, 3), dtype=np.uint8)  # 3 channels for RGB

for z in range(total_z):
    z_slice = image[z, 0, :, :]
    binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    
    

# Extract z slice and channel from raw image file
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

# Display image
def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def categorize_pixels(channel_images):
    categorized_image = np.zeros((x_pixels, y_pixels, 3), dtype=np.uint8)  # 3 channels for RGB

    for x in range(x_pixels):
        for y in range(y_pixels):
            # Get the pixel values for the three channels
            channel_0_value = channel_images[0][x, y]
            channel_1_value = channel_images[1][x, y]
            channel_2_value = channel_images[2][x, y]

            if channel_0_value < 50:
                # Black for pixels where channel 0 value is less than 10
                categorized_image[x, y] = (0, 0, 0)
            elif channel_1_value > channel_2_value and channel_0_value < 3 * channel_1_value:
                # Red for pixels where channel 1 is higher than channel 2,
                # unless channel 0 is 2x or greater than channel 1
                categorized_image[x, y] = (0, 0, 255)
            elif channel_2_value > channel_1_value and channel_0_value < 3 * channel_2_value:
                # Green for pixels where channel 2 is higher than channel 1,
                # unless channel 0 is 2x or greater than channel 2
                categorized_image[x, y] = (0, 255, 0)
            else:
                # Blue for remaining pixels
                categorized_image[x, y] = (255, 0, 0)

    # Display the categorized image
    #display_image(categorized_image, "Categorized Image")

    # Optionally, return the categorized image if further processing is needed
    return categorized_image

def process_z_stack(image, total_z, total_channels):
    # Initialize an array to store the categorized images for each z-slice
    z_stack = np.zeros((total_z, x_pixels, y_pixels, 3), dtype=np.uint8)  # 3 channels for RGB

    # Iterate over each z-slice
    for z_slice in range(total_z):
        # Extract the z-slice for each channel
        channel_images = mask_each_channel(image, z_slice)
        
        # Categorize pixels for this z-slice
        categorized_image = categorize_pixels(channel_images)
        
        # Store the categorized image in the z-stack
        z_stack[z_slice] = categorized_image

    # Save the z-stack as a TIFF file
    #tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/gastruloid_categorized.tiff', z_stack)

    #print("Z-stack saved successfully.")
            

process_z_stack(image, total_z, total_channels)

print("\n")
print("Complete")
