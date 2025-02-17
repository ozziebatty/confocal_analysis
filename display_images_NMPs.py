from datetime import datetime
import numpy as np
import tifffile
import napari

print(f"{datetime.now():%H:%M:%S} - Importing Napari...")

image_pathway = '/Users/oskar/Desktop/translation_NMPs_image_analysis/NM_SBSE_OPP_d5_2024_10_28__17_40_50__p5.tiff'
image = tifffile.imread(image_pathway)
print(image.shape)
total_channels = image.shape[1]

channel_names = ["Sox1", "Bra", "Sox2", "OPP"]

z_slice_to_use = {
    'Bra': 16,
    'Sox1': 16,
    'OPP':  16,
    'Sox2': 16
}

colours = {
    'Sox1': [150, 150, 0],    # Green
    'Bra': [220, 0, 0],     # Red
    'Sox2': [60, 115, 255],    # Blue
    'OPP': [160, 0, 160]    # Purple
}

contrast_limits = {
    'Sox1': (10, 90),
    'Bra': (0, 255),
    'Sox2': (0, 255),
    'OPP': (25, 105)
}

gamma_values = {
    'Sox1': 1.1,
    'Bra': 1,
    'Sox2': 1,
    'OPP': 0.7
}


viewer = napari.Viewer()

# Add each channel as a separate layer with its own RGB color
for channel_index in range(total_channels):
    channel = channel_names[channel_index]
    if not channel == 'Sox2':
        z_to_use = z_slice_to_use[channel]
        rgb_color = colours[channel]
        contrast_min, contrast_max = contrast_limits[channel]
        gamma = gamma_values[channel]

        # Apply single channel parameters
        single_channel = image[:, channel_index, :, :].astype(np.uint8)

        # Create an RGB image by stacking the single-channel image
        rgb_image = np.zeros((single_channel.shape[0], single_channel.shape[1], single_channel.shape[2], 3), dtype=np.uint8)
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