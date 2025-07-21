import tifffile
import skimage
import numpy as np
import napari

scale_factor = 2

flipped_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_new_downside_image_single_channel.tiff'
new_upside_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_new_upside_image_single_channel.tiff'
upside_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_complete.tiff'

upside_image = tifffile.imread(upside_image_pathway)


'''
image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_downside.tiff'
image = tifffile.imread(image_pathway)

print(np.max(image))

print("Converting from uint12 to uint8")
converted_image = (image / 4095 * 255).astype(np.uint8)
image = converted_image

tifffile.imwrite('/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_downside_converted.tiff', image)

image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_downside_converted.tiff'
'''

def downsize_image(image, scale_factor):
    # Calculate new shape
    total_z, total_channels, y_pixels, x_pixels = image.shape

    print("Original image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

    new_shape = (total_z, total_channels, y_pixels // scale_factor, x_pixels // scale_factor)
    
    # Resize image
    downsized_image = np.zeros(new_shape, dtype=np.uint16)

    for z in range(total_z):
        z_slice = image[z]

        for channel in range(total_channels):
            channel_z_slice = z_slice[channel]

            downsized_channel_z_slice = (skimage.transform.resize(channel_z_slice, (y_pixels // scale_factor, x_pixels // scale_factor), anti_aliasing=True, preserve_range=True))

            downsized_image[z][channel] = downsized_channel_z_slice

    print("Downsized image shape (z, c, y, x) :", downsized_image.shape, "dtype:", image.dtype)

    return downsized_image

#downsized_image = downsize_image(image, scale_factor)

image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_downside_downsized_converted.tiff'
image = (tifffile.imread(image_pathway).astype(np.uint8))


def flip_image(image):
    z_flipped_image = np.flip(image, axis = 0) #Flip z
    flipped_image = np.flip(z_flipped_image, axis = 2) #Flip y

    return flipped_image

def extract_channel(image, channel):

upside_image = extract_channel(image, 0)


flipped_image = flip_image(image.astype(np.uint8))

#tifffile.imwrite(flipped_image_pathway, flipped_image, imagej=True, dtype=np.uint8)  # Prevent type conversion)

#tifffile.imwrite(new_upside_image_pathway, upside_image, imagej=True, dtype=np.uint8)  # Prevent type conversion)


viewer = napari.Viewer()
viewer.add_image(upside_image)
viewer.add_image(flipped_image)
#viewer.add_image(downsized_image)
#viewer.add_image(flipped_image)
napari.run()