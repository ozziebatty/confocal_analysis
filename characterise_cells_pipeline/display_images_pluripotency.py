from datetime import datetime
import numpy as np
import tifffile
import napari

print(f"{datetime.now():%H:%M:%S} - Importing Napari...")

image_pathway = '/Users/oskar/Desktop/OPP_BS2_Oct4/cropped/file_2_e.tiff'
image = tifffile.imread(image_pathway)
print(image.shape)
total_channels = image.shape[1]

channel_names = ["Oct4", "Bra", "Sox2", "OPP"]

z_slice_to_use = {
    'Oct4': 0,
    'Sox2': 1,
    'OPP':  1
}

colours = {
    'Oct4': [30, 200, 30],    # Green
    'Bra': [255, 0, 0],     # Red
    'Sox2': [60, 115, 255],    # Blue
    'OPP': [180, 0, 140]    # Purple
}

contrast_limits = {
    'Oct4': (95, 210),
    'Bra': (0, 220),
    'Sox2': (25, 140),
    'OPP': (25, 105)
}

gamma_values = {
    'Oct4': 1.2,
    'Bra': 1.5,
    'Sox2': 1.4,
    'OPP': 0.7
}


viewer = napari.Viewer()

# Add each channel as a separate layer with its own RGB color
for channel_index in range(total_channels):
    channel = channel_names[channel_index]
    if not channel == 'Bra':
        z_to_use = z_slice_to_use[channel]
        rgb_color = colours[channel]
        contrast_min, contrast_max = contrast_limits[channel]
        gamma = gamma_values[channel]

        # Apply single channel parameters
        single_channel = image[z_to_use, channel_index, :, :].astype(np.uint8)

        # Create an RGB image by stacking the single-channel image
        rgb_image = np.zeros((single_channel.shape[0], single_channel.shape[1], 3), dtype=np.uint8)
        for i in range(3):  # Apply the RGB color to each channel
            rgb_image[..., i] = single_channel * (rgb_color[i] / 255.0)

        viewer.add_image(
            rgb_image,
            name=channel,
            contrast_limits=contrast_limits[channel],
            rgb=True,  # Indicate that this is an RGB image
            blending='additive',  # Use additive blending for transparency
            opacity=0.7,  # Adjust opacity as needed
            gamma=gamma
        )

napari.run()