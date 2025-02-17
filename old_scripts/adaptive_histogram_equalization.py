import numpy as np
import tifffile
import pandas as pd
from skimage.exposure import equalize_adapthist
from skimage.transform import rescale
from skimage import img_as_ubyte

# Read an image from file
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')

# Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]

# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
x_pixels = image.shape[2]
y_pixels = image.shape[3]

# Apply adaptive histogram equalization
equalized_stack = np.zeros_like(image)

# Kernel size for equalization, adjust based on object size
kernel_size = (10, 10)  # You may also consider tuning this based on the object size

equalized_stack = equalize_adapthist(image)

channel_image_zero = image[:, 0, :, :]
channel_image_one = image[:, 1, :, :]
channel_image_two = image[:, 2, :, :]
channel_image_three = image[:, 3, :, :]

# Replace these with the paths to your 3D Z stack image files
image_channels = [channel_image_zero, channel_image_one, channel_image_two, channel_image_three]

# Read the 3D Z stacks into numpy arrays

# Convert the list of Z stacks into a single 5D array (Z, C, H, W)
# Stack the 3D arrays along the channel dimension
stacked_images = np.stack(image_channels, axis=3)

# Save the stack as a single multi-channel TIFF
#tifffile.imwrite(output_path, equalized_stack.astype(np.float32))
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/channel_test.tiff', stacked_images.astype(np.uint8))
#tifffile.imwrite(output_path, channel_image.astype(np.uint8))

image_test = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/channel_test.tiff')
print(image_test.shape)

#equalized_first_stack = equalize_adapthist(channel_image_zero[0], kernel_size=None, clip_limit=0.01, nbins=256)

equalized_first_stack = np.zeros_like(channel_image_zero)

equalized_test_DAPI = np.zeros_like(channel_image_zero)
print("before", equalized_test_DAPI.shape)

for z in range(total_z):
    # Apply adaptive histogram equalization on each z-slice
    equalized_slice = equalize_adapthist(channel_image_zero[z], kernel_size=None, clip_limit=0.01, nbins=256)
    equalized_test_DAPI[z] = img_as_ubyte(equalized_slice)

print("equalized_test_DAPI 0 ", equalized_test_DAPI[0])
print("channel images 0 ", channel_image_zero[0])
print("after", equalized_test_DAPI.shape)

    
    
print("after", equalized_test_DAPI.shape)

tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/test_DAPI.tiff', channel_image_zero)
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/equalized_test_DAPI.tiff', equalized_test_DAPI)
tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/first_stack.tiff', equalized_first_stack)



print("\nComplete")
