import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
import pandas as pd

# Read an image from file
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/gastruloid_z.lsm')

normalize = True

#Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]

# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
x_pixels = image.shape[2]
y_pixels = image.shape[3]
z_slice = 180
masked_pixels = 0

# Extract z slice and channel from image
def image_snap(image, z_slice, channel):
    working_image = image[z_slice, channel, :, :]
    working_image_normalized = cv2.normalize(working_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return working_image_normalized

def create_mask(image, z_slice):
    global masked_pixels
    
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
    masked_pixels = np.sum(mask > 0)

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

def normalize_image_to_reference(reference_image, image_to_normalize):
    with np.errstate(divide='ignore', invalid='ignore'):
        # Normalize each pixel value in image_to_normalize by the corresponding pixel value in reference_image
        normalized_image = np.divide(image_to_normalize, reference_image, where=(reference_image > 0))
        normalized_image[np.isnan(normalized_image)] = 0  # Replace NaNs with 0 if any
        display_image(normalized_image, 'title') 
    return normalized_image

# Calculate average fluorescence in the masked region
def calculate_average_fluorescence(channel_images, masked_pixels, normalize, reference_channel):
    average_fluorescence_all_channels = [None]*total_channels
    for i in range(len(channel_images)):
        image = channel_images[i]
        if normalize == True:
            reference_image = channel_images[reference_channel]
            image = normalize_image_to_reference(reference_image, image)
        total_fluorescence = np.sum(image)
        average_fluorescence = total_fluorescence / masked_pixels if masked_pixels > 0 else 0
        average_fluorescence_all_channels[i] = average_fluorescence     
    return average_fluorescence_all_channels

# Display image
def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


average_fluorescence = []
for z_slice in range(total_z):
    masked_channel_images = mask_each_channel(image, z_slice)
    average_fluorescence_all_channels = calculate_average_fluorescence(masked_channel_images, masked_pixels, normalize, 0)
    average_fluorescence.append(average_fluorescence_all_channels)

df = pd.DataFrame(average_fluorescence, columns=channel_names)
print(df)

if normalize == True:
    save_file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/normalized_average_fluorescence.csv'
else:
    save_file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/average_fluorescence.csv'

df.to_csv(save_file_path, index=False)
    

print("\n")
print("Complete")
