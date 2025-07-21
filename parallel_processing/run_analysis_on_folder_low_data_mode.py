from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - IMPORTING PACKAGES...")
start_time = datetime.now()

import subprocess
import sys
import pandas as pd
import os
import shutil
import multiprocessing

display_napari = 'False'

input_folder = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_control'
# input_folder = '/Volumes/steventon/Users/Oskar/Confocal/SBSE_control_BMH21/control'
output_folder = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis_thresholds'

#original_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_semi_cropped.tiff'

channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
dapi_channel = 3

def process_folder(input_folder):

    # Iterate over each file in the input directory
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        cpu_num = 0

        if file_name.endswith(('.tiff')) and os.path.isfile(input_path):
            print("\n")

            print(f"RUNNING ANALYSIS ON {file_name}...")

            original_image_pathway = input_path

            run_pipeline(input_path, output_folder, file_name, cpu_num, dapi_channel, display_napari)

            cpu_num += 1

def create_folder(folder_pathway):
    # Check if the folder already exists
    if not os.path.exists(folder_pathway):
        os.makedirs(folder_pathway)  # Create the folder

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

def run_pipeline(input_path, output_folder, file_name, cpu_num, dapi_channel, display_napari):
    image_start_time = datetime.now()

    print("\n")
    
    print(f"{datetime.now():%H:%M:%S} - Preprocessing...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/preprocess.py', input_path, output_folder, cpu_num, dapi_channel, display_napari)     
          
    print("\n")

    print(f"{datetime.now():%H:%M:%S} - Segmenting...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/cellpose_segment_2D.py', output_folder, cpu_num, dapi_channel, display_napari)
          
    print("\n")

    print(f"{datetime.now():%H:%M:%S} - Stitching...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/stitch.py', output_folder, cpu_num, dapi_channel, display_napari)
   
    print("\n")

    print(f"{datetime.now():%H:%M:%S} - Characterising cells...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/characterise_cells.py', output_folder, file_name, cpu_num, dapi_channel, display_napari)


    print("\n")


    image_end_time = datetime.now()
    image_time_taken = image_end_time - image_start_time
    print(f"{datetime.now():%H:%M:%S} - Analysis complete in:", str(image_time_taken).split('.')[0])
    print("\n")

create_folder(output_folder)

process_folder(input_folder)

end_time = datetime.now()
time_taken = end_time - start_time
print("Total time taken:", str(time_taken).split('.')[0])