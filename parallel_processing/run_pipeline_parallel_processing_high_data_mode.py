import multiprocessing
from multiprocessing import Pool
from datetime import datetime
print(f"{datetime.now():%H:%M:%S} - IMPORTING PACKAGES...")
start_time = datetime.now()
import subprocess
import sys
import pandas as pd
import os
import shutil

display_napari = 'False'
#input_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSO_colour_control/results'
#output_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSO_colour_control/results'
input_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results'
output_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results'

channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
dapi_channel = 3

threshold_folder = output_folder

def process_folder(input_folder):
    # Get list of all .tiff files
    files_to_process_original_pathways = [
        os.path.join(input_folder, file) 
        for file in os.listdir(input_folder) 
        if file.endswith('.tiff') and os.path.isfile(os.path.join(input_folder, file))
    ]

    files_to_process_updated_pathways = [create_folder_and_update_pathway(file) for file in files_to_process_original_pathways]
    
    # Get number of available CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"\nProcessing {len(files_to_process_updated_pathways)} files using {num_cpus} CPUs")
    
    # Create list of (file, cpu_num) tuples for processing
    # CPU numbers will cycle through available CPUs
    process_args = [(file, i % num_cpus) for i, file in enumerate(files_to_process_updated_pathways)]
    
    # Create pool and process files in parallel
    with Pool(processes=num_cpus) as pool:
        pool.map(process_image, process_args)

def create_folder_and_update_pathway(original_image_pathway):
    """
    Creates a folder from the image (if it doesn't exist) at the same location as the image
    and saves the original image into the new folder.
    
    :param image_path: The full path to the image file.
    """

    # Extract file name without extension
    image_name = os.path.splitext(os.path.basename(original_image_pathway))[0]  # Extract file name without extension
    sub_folder_pathway = os.path.join(output_folder, image_name)  # Combine output folder and image name as sub-folder path

    # Check if the folder already exists
    if not os.path.exists(sub_folder_pathway):
        os.makedirs(sub_folder_pathway)  # Create the folder

    # Define the new image pathway
    new_image_pathway = os.path.join(sub_folder_pathway, "original.tiff")

    # Copy the image file to the new folder
    shutil.copy(original_image_pathway, new_image_pathway)

    return new_image_pathway

def process_folder_of_folders(input_folder):
    sub_folders_to_process = [
        os.path.join(input_folder, folder) for folder in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, folder))
    ]

    print(sub_folders_to_process)

    # Get number of available CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"\nProcessing {len(sub_folders_to_process)} files using {num_cpus} CPUs")

    # Create list of (file, cpu_num) tuples for processing
    # CPU numbers will cycle through available CPUs
    process_args = [(sub_folder, i % num_cpus) for i, sub_folder in enumerate(sub_folders_to_process)]
    
    # Create pool and process files in parallel
    with Pool(processes=num_cpus) as pool:
        pool.map(process_sub_folder, process_args)

def process_sub_folder(args):
    """Process a single file with its CPU number"""
    file_path, cpu_num = args
    sub_output_folder = file_path
    return run_pipeline(file_path, sub_output_folder, threshold_folder, cpu_num, dapi_channel, display_napari)

def process_image(args):
    """Process a single file with its CPU number"""
    file_path, cpu_num = args
    file_name = os.path.basename(file_path)
    sub_output_folder = os.path.dirname(file_path)
    #print(f"\nRUNNING ANALYSIS ON {file_name} using CPU {cpu_num}...")
    return run_pipeline(file_path, sub_output_folder, threshold_folder, cpu_num, dapi_channel, display_napari)

def run_pipeline(input_path, sub_output_folder, threshold_folder, cpu_num, dapi_channel, display_napari):
    image_start_time = datetime.now()
    
    '''
    print(f"\n{datetime.now():%H:%M:%S} - Preprocessing on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/preprocess.py', 
            input_path, sub_output_folder, cpu_num, dapi_channel, display_napari)
    
    print(f"\n{datetime.now():%H:%M:%S} - Segmenting on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/cellpose_segment_2D.py', 
            sub_output_folder, cpu_num, dapi_channel, display_napari)
    
    print(f"\n{datetime.now():%H:%M:%S} - Stitching on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/stitch.py', 
            sub_output_folder, cpu_num, dapi_channel, display_napari)
    
    
    

    print(f"\n{datetime.now():%H:%M:%S} - Finding bleedthrough coefficients {cpu_num}...")
    run_script('//Users/oskar/Desktop/steventon_lab/image_analysis/scripts/find_contributions_three.py', 
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)

    
    print(f"\n{datetime.now():%H:%M:%S} - Deconvoluting on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/demultiplexing.py', 
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)   


    print(f"\n{datetime.now():%H:%M:%S} - Finding thresholds on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/find_segmented_thresholds.py', 
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)
    
    
    print(f"\n{datetime.now():%H:%M:%S} - Characterising cells on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/characterise_cells_by_threshold.py', 
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)

    
    print(f"\n{datetime.now():%H:%M:%S} - Displaying image on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/display_images.py',
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)
    '''
    print(f"\n{datetime.now():%H:%M:%S} - Assigning fates on CPU {cpu_num}...")
    run_script('/Users/oskar/Desktop/steventon_lab/image_analysis/scripts/characterise_cells_pipeline/assign_fates_by_intensity.py',
            sub_output_folder, threshold_folder, cpu_num, channel_names, dapi_channel, display_napari)

    
    image_end_time = datetime.now()
    image_time_taken = image_end_time - image_start_time
    #print(f"\n{datetime.now():%H:%M:%S} - Analysis complete on CPU {cpu_num} in:", str(image_time_taken).split('.')[0])
    print("\n")

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


if __name__ == "__main__":
    #process_folder(input_folder)
    process_folder_of_folders(input_folder)
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Total time taken:", str(time_taken).split('.')[0])