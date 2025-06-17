#%%
import numpy as np
import tifffile
import napari
import csv
import os
import matplotlib.pyplot as plt

#%%
# File paths
image_index = 7
tiff_file_path = "C:/Users/ob361/data/chick_OPP/HH8/max/7.tif"
masks_path = "C:/Users/ob361/data/chick_OPP/HH8/masks/mask_7.npy"
csv_path = "C:/Users/ob361/data/chick_OPP/HH8/region_intensities.csv"

#%%
# --- Part 1: Draw 3 regions and save masks ---

image = tifffile.imread(tiff_file_path)  # shape: (c, y, x) after max projection
background_channel = image[0]  # channel 0 for drawing, shape: (y, x)

background_channel_brightened = background_channel * 2

labels = np.zeros_like(background_channel, dtype=np.uint8)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(background_channel, name='channel_0', colormap='gray')
    label_layer = viewer.add_labels(labels, name='regions')

    @viewer.bind_key('s')
    def save_mask(viewer):
        np.save(masks_path, label_layer.data.astype(np.uint8))
        print("masks saved")

#%%
# --- Part 2: Load masks and calculate mean intensity of channel 3 ---

image = tifffile.imread(tiff_file_path)  # shape: (c, y, x)
masks = np.load(masks_path)  # shape: (y, x)

channel_1 = image[1]  # shape: (y, x)

rows = []
print(image_index)

for region in [1, 2, 3, 4]:
    region_mask = masks == region  # boolean mask
    mean_intensity = channel_1[region_mask].mean()
    rows.append((image_index, region, mean_intensity))
    print(f"Region {region}: mean intensity = {mean_intensity:.2f}")

write_header = not os.path.exists(csv_path)

with open(csv_path, 'a', newline='') as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(['image_index', 'region', 'translation'])
    writer.writerows(rows)

print("done")

#%%
# Prepare data structure: region -> list of intensities
x = []
y = []

with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        x.append(int(row['region']))
        y.append(float(row['translation']))

plt.figure(figsize=(5, 4))
plt.scatter(x, y, s=20)

plt.xlabel('Region')
plt.ylabel('Translation intensity')
plt.title('Translation per region')
plt.xticks(sorted(set(x)))
plt.tight_layout()
plt.show()
# %%
