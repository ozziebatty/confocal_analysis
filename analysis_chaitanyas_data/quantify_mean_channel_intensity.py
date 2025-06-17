import tifffile
import numpy as np
import os
import csv

# Set top-level directory and output
base_dir = r"C:\Users\ob361\data\glucose_Bra_Sox2\Day 3 - 4\all_images_d3_to_d4"
output_csv_path = r"C:\Users\ob361\data\glucose_Bra_Sox2\masked_channel_intensity_means.csv"

days = '2-5'  # Set the days variable to the desired value

# CSV header
header = [
    "image_name", "inverse", "treatment", "set", "replicate", "days",
    "channel_0", "channel_1", "channel_2", "channel_3"
]
write_header = not os.path.exists(output_csv_path)

# Utility to infer replicate from filename like "No_A.tif" or "iNo_A.tif"
def get_replicate(filename):
    base = os.path.splitext(filename)[0]
    clean = base[1:] if base.startswith("i") else base
    parts = clean.split("_")
    return parts[-1] if len(parts) > 1 else "?"

# Traverse and analyse
with open(output_csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(header)

    for root, _, files in os.walk(base_dir):
        for filename in files:
            if not filename.lower().endswith(".tif"):
                continue
            if filename.startswith("._"):
                continue  # Skip macOS metadata files

            image_path = os.path.join(root, filename)
            inverse = filename.startswith("i")
            replicate = get_replicate(filename)

            # Keep full filename as image_name
            image_name = filename  

            # Get treatment and set from folder structure
            rel_path = os.path.relpath(image_path, base_dir)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) < 3:
                print(f"Skipping {filename} — not deep enough in folder tree")
                continue

            treatment = path_parts[0]
            set_name = path_parts[1]

            print(f"Processing: {filename} | Treatment: {treatment} | Set: {set_name} | Replicate: {replicate}")

            with tifffile.TiffFile(image_path) as tif:
                image = tif.asarray()

            if image.ndim != 4:
                raise ValueError(f"{filename}: Expected 4D image (Z, C, Y, X), got {image.shape}")
            z, c, y, x = image.shape
            if c != 4:
                raise ValueError(f"{filename}: Expected 4 channels, got {c}")
            if image.dtype != np.uint8:
                raise ValueError(f"{filename}: Expected 8-bit image, got {image.dtype}")

            dapi_mask = image[:, 0, :, :] > 20

            channel_means = [(image[:, i, :, :][dapi_mask]).mean() for i in range(c)]
            row = [image_name, inverse, treatment, set_name, replicate, days] + channel_means
            writer.writerow(row)


print("All images analysed and saved.")
