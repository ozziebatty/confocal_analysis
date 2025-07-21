from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import numpy as np
import tifffile
import napari
import csv
import pandas as pd
import sys
import os

debug_mode = False

if debug_mode == True:
    output_folder_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis'
    file_name = 'test'
    DAPI_channel = 2
    remove_channel = 'False'
    channel_to_remove = 2
    display_napari = 'False'
else:
    output_folder_pathway = sys.argv[1]
    #print(output_folder_pathway)
    file_name = sys.argv[2]
    DAPI_channel = int(sys.argv[3])
    remove_channel = sys.argv[4]
    channel_to_remove = 2
    display_napari = sys.argv[5]

#print(f"{datetime.now():%H:%M:%S} - Loading image...")

image_pathway = output_folder_pathway + '/preprocessed.tiff'
image = tifffile.imread(image_pathway)

if remove_channel == 'True':
    image = image[:, np.delete(np.arange(image.shape[1]), channel_to_remove), :, :]

total_channels = image.shape[1]
#print("Image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)

threshold = 50

results = []

DAPI_channel_slice = image[:, DAPI_channel, :, :]
DAPI_channel_slice = DAPI_channel_slice.flatten()
non_zero_DAPI_channel_slice = DAPI_channel_slice[DAPI_channel_slice > 0]
total_DAPI_intensity = np.sum(non_zero_DAPI_channel_slice)

for channel in range(total_channels):
    channel_slice = image[:, channel, :, :]
    channel_slice = channel_slice.flatten()

    relative_channel_slice = channel_slice / DAPI_channel_slice
    relative_channel_slice_cleaned = relative_channel_slice[(relative_channel_slice > 0) & (relative_channel_slice < 256)]
    relative_pixel_intensity = np.mean(relative_channel_slice_cleaned)

    non_zero_channel_slice = channel_slice[channel_slice > 0]
    channel_intensity = np.mean(non_zero_channel_slice)
    pixels_over_threshold = len(non_zero_channel_slice[non_zero_channel_slice > threshold])
    relative_channel_intensity = np.sum(non_zero_channel_slice) / total_DAPI_intensity

    # Append the data to the results list
    results.append({
        "name": file_name,
        "condition": 'treated',
        "channel": channel,
        "relative_pixel_intensity": relative_pixel_intensity,
        "channel_intensity": channel_intensity,
        "pixels_over_threshold": pixels_over_threshold,
        "relative_channel_intensity": relative_channel_intensity
    })

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

data_pathway = output_folder_pathway + '/' + file_name + '_data.csv'

try:
    # Attempt to save the DataFrame to a CSV file
    df.to_csv(data_pathway, index=False)
    print(f"{datetime.now():%H:%M:%S} - Results saved successfully at {data_pathway}")
except FileNotFoundError:
    print(f"{datetime.now():%H:%M:%S} - Error: The folder '{os.path.dirname(data_pathway)}' does not exist.")
except PermissionError:
    print(f"{datetime.now():%H:%M:%S} - Error: Permission denied when trying to save to '{data_pathway}'.")
except Exception as e:
    print(f"{datetime.now():%H:%M:%S} - Unexpected error: {e}")