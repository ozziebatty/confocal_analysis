#%%
import numpy as np
import tifffile
import napari
import csv

# File paths
tiff_file_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris_cropped_preprocessed.tiff"
masks_path = "/Users/oskar/Desktop/format_test/masks/mask_a.npy"
csv_path = "/Users/oskar/Desktop/format_test/region_intensities.csv"

#%%
# --- Part 1: Draw 3 regions and save masks ---

image = tifffile.imread(tiff_file_path)  # shape: (z, c, y, x)
background_channel = image[:, 0]  # channel 0 for drawing, shape: (z, y, x)

labels = np.zeros_like(background_channel, dtype=np.uint8)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(background_channel, name='channel_0', colormap='gray')
    label_layer = viewer.add_labels(labels, name='regions (1-3)')

    @viewer.bind_key('s')
    def save_mask(viewer):
        np.save(masks_path, label_layer.data.astype(np.uint8))
        print("masks saved")

#%%
# --- Part 2: Load masks, project to 2D, and calculate mean intensity of channel 3 ---

image = tifffile.imread(tiff_file_path)  # shape: (z, c, y, x)
masks = np.load(masks_path)  # shape: (z, y, x)

# Project the 3D masks down to 2D using maximum projection (keep largest label at each (y, x) point)
masks_projected = masks.max(axis=0)  # shape: (y, x)

channel_3 = image[:, 3]  # shape: (z, y, x)

rows = []
for region in [1, 2, 3, 4]:
    # Broadcast the 2D region mask across all z-slices
    region_mask = masks_projected == region  # shape: (y, x)
    region_mask_3d = np.broadcast_to(region_mask, channel_3.shape)  # shape: (z, y, x)
    
    mean_intensity = channel_3[region_mask_3d].mean()
    rows.append((region, mean_intensity))
    print(f"Region {region}: mean intensity = {mean_intensity:.2f}")

with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['region', 'mean_intensity_ch3'])
    writer.writerows(rows)

print("done")

# %%
