from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - IMPORTING PACKAGES...")
start_time = datetime.now()
import subprocess
import sys
import pandas as pd
import os
import shutil
import multiprocessing
from multiprocessing import Pool

display_napari = 'False'
input_folder = '/Volumes/steventon/Users/Oskar/Confocal/SBSE_control_BMH21/control'
# input_folder = '/Volumes/steventon/Users/Oskar/Confocal/SBSE_control_BMH21/control'
output_folder = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis_thresholds'
channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
dapi_channel = 3

def process_single_file(args):
    """Process a single file with its CPU number"""
    file_path, cpu_num = args
    file_name = os.path.basename(file_path)
    print(f"\nRUNNING ANALYSIS ON {file_name} using CPU {cpu_num}...")
    return run_pipeline(file_path, output_folder, file_name, cpu_num, dapi_channel, display_napari)

def process_folder(input_folder):
    # Get list of all .tiff files
    files_to_process = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.endswith('.tiff') and os.path.isfile(os.path.join(input_folder, f))
    ]
    
    # Get number of available CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"\nProcessing {len(files_to_process)} files using {num_cpus} CPUs")
    
    # Create list of (file, cpu_num) tuples for processing
    # CPU numbers will cycle through available CPUs
    process_args = [(f, i % num_cpus) for i, f in enumerate(files_to_process)]
    
    # Create pool and process files in parallel
    with Pool(processes=num_cpus) as pool:
        pool.map(process_single_file, process_args)

def create_folder(folder_pathway):
    # Check if the folder already exists
    if not os.path.exists(folder_pathway):
        os.makedirs(folder_pathway) # Create the folder
    return folder_pathway

def run_script(script, *args):
    # Convert arguments to strings (supporting arrays, image paths, or strings) (-u logs in real time)
    args = [str(arg) for arg in args]
    command = ["/usr/local/bin/python3", "-u", script] + args
    try:
        # Run the script
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, # Pipe the standard output
            stderr=subprocess.PIPE, # Pipe the standard error
            text=True # Ensure output is returned as a string
        )
        # Stream the output in real time
        for line in process.stdout:
            print(line, end='') # Print each line as it is received
        # Wait for the process to complete
        process.wait()
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e.stderr}")
        raise RuntimeError(f"Script {script} failed.") from e

def run_pipeline(input_path, output_folder, file_name, cpu_num, dapi_channel, display_napari):
    image_start_time = datetime.now()
    print(f"\n{datetime.now():%H:%M:%S} - Preprocessing on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/preprocess.py', 
               input_path, output_folder, cpu_num, dapi_channel, display_napari)
    
    print(f"\n{datetime.now():%H:%M:%S} - Segmenting on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/cellpose_segment_2D.py', 
               output_folder, cpu_num, dapi_channel, display_napari)
    
    print(f"\n{datetime.now():%H:%M:%S} - Stitching on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/stitch.py', 
               output_folder, cpu_num, dapi_channel, display_napari)
    
    print(f"\n{datetime.now():%H:%M:%S} - Characterising cells on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/characterise_cells.py', 
               output_folder, file_name, cpu_num, dapi_channel, display_napari)
    
    image_end_time = datetime.now()
    image_time_taken = image_end_time - image_start_time
    print(f"\n{datetime.now():%H:%M:%S} - Analysis complete on CPU {cpu_num} in:", str(image_time_taken).split('.')[0])
    print("\n")

if __name__ == '__main__':
    create_folder(output_folder)
    process_folder(input_folder)
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Total time taken:", str(time_taken).split('.')[0])