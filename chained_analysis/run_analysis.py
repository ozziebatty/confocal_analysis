import subprocess
import sys
import pandas as pd
import os
import shutil
from datetime import datetime

import napari
viewer = napari.Viewer()

display_napari = True

original_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_semi_cropped.tiff'

dapi_channel = 0
channel_names = ["Sox1", "Sox2", "DAPI", "Bra"]


def create_folder(original_image_pathway):
    """
    Creates a folder from the image (if it doesn't exist) at the same location as the image
    and saves the original image into the new folder.
    
    :param image_path: The full path to the image file.
    """
    
    # Extract folder path and file name from the image path
    folder_pathway = os.path.splitext(original_image_pathway)[0]  # Remove file extension for folder name
    image_name = os.path.basename(original_image_pathway)  # Get the image file name
    
    # Check if the folder already exists
    if not os.path.exists(folder_pathway):
        os.makedirs(folder_pathway)  # Create the folder
        

    new_image_pathway = os.path.join(folder_pathway, 'original.tiff')

    # Copy the image file to the new folder
    shutil.copy(original_image_pathway, new_image_pathway)
    
    return folder_pathway

def run_script(script, *args):

    # Convert arguments to strings (supporting arrays, image paths, or strings) (-u logs in real time)
    args = [str(arg) for arg in args]

    command = ["/usr/local/bin/python3", "-u", script] + args
    
    try:
        # Run the script
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE,  # Pipe the standard output
            stderr=subprocess.PIPE,  # Pipe the standard error
            text=True                # Ensure output is returned as a string
        )

        # Stream the output in real time
        for line in process.stdout:
            print(line, end='')  # Print each line as it is received

        # Wait for the process to complete
        process.wait()

    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e.stderr}")
        raise RuntimeError(f"Script {script} failed.") from e

def run_pipeline(image_file, dapi_channel, display_napari):
    print("\n")
    
    print(f"{datetime.now():%H:%M:%S} - Preprocessing...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/preprocess.py', folder_pathway, dapi_channel, display_napari)
    
    print("\n")
    print(f"{datetime.now():%H:%M:%S} - Segmenting...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/cellpose_segment_2D.py', folder_pathway, dapi_channel, display_napari)

    print("\n")
    print(f"{datetime.now():%H:%M:%S} - Stitching...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/stitch.py', folder_pathway, dapi_channel, display_napari)
    
    print("\n")
    print(f"{datetime.now():%H:%M:%S} - Calculating fluorescence values...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/calculate_fluorescence_values.py', folder_pathway, dapi_channel, display_napari)

    print("\n")
    print(f"{datetime.now():%H:%M:%S} - Complete")
    print("\n")

folder_pathway = create_folder(original_image_pathway)

run_pipeline(folder_pathway, dapi_channel, display_napari)
