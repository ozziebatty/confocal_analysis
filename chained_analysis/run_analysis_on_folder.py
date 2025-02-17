from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - IMPORTING PACKAGES...")
start_time = datetime.now()

import subprocess
import sys
import pandas as pd
import os
import shutil

import napari
viewer = napari.Viewer()

display_napari = False

input_folder = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis'

#original_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_semi_cropped.tiff'

dapi_channel = 2
channel_names = ["Sox1", "Sox2", "DAPI", "Bra"]

def process_folder(input_folder):

    # Iterate over each file in the input directory
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # Check if it's an image file (adjust as needed for your format, e.g., '.tif', '.png')
        if file_name.endswith(('.tif', '.png', '.jpg', '.tiff')) and os.path.isfile(input_path):
            print("\n")

            print(f"PROCESSING {file_name}...")

            original_image_pathway = input_path

            subfolder_pathway = create_folder(original_image_pathway)

            run_pipeline(subfolder_pathway, file_name, dapi_channel, display_napari)

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

def run_pipeline(folder_pathway, file_name, dapi_channel, display_napari):
    image_start_time = datetime.now()

    print("\n")
    
    print(f"{datetime.now():%H:%M:%S} - Preprocessing...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/preprocess.py', folder_pathway, dapi_channel, display_napari)     
          
    print("\n")

    print(f"{datetime.now():%H:%M:%S} - Calculating fluorescence values...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/quantify_average_channel_intensity.py', folder_pathway, file_name, dapi_channel, 'True', display_napari)

    print("\n")
    image_end_time = datetime.now()
    image_time_taken = image_end_time - image_start_time
    print(f"{datetime.now():%H:%M:%S} - Analysis complete in:", str(image_time_taken).split('.')[0])
    print("\n")

process_folder(input_folder)

end_time = datetime.now()
time_taken = end_time - start_time
print("Total time taken:", str(time_taken).split('.')[0])