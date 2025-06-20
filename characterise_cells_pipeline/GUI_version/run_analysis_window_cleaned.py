print("Importing packages...")
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
import tifffile
import threading
from cellpose import models
import napari


#%%
print("Initialising window...")
preprocessed_image = None
segmented_image = None
root_window = None
loaded_project = True
progress_dialog = None

class progressdialog:
    def __init__(self, parent, title, total_images):
        self.parent = parent
        self.total_images = total_images
        self.current_image = 0
        self.cancelled = False
        
        # Create a new top-level window
        self.root = tk.Toplevel(parent)
        self.root.title(title)
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        self.root.transient(parent)  # Make dialog modal
        self.root.grab_set()  # Make dialog modal
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close
        
        # Create widgets
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Overall image progress
        ttk.Label(main_frame, text="Overall Progress").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.image_progress_text = ttk.Label(main_frame, text=f"Image 0/{self.total_images}")
        self.image_progress_text.grid(row=0, column=1, sticky=tk.E, pady=(0, 5))
        
        self.image_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=self.total_images)
        self.image_progress.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 15))
        
        # Preprocessing progress
        ttk.Label(main_frame, text="Preprocessing").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        #self.preprocess_text = ttk.Label(main_frame, text="0%")
        #self.preprocess_text.grid(row=2, column=1, sticky=tk.E, pady=(0, 5))
        
        self.preprocess_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.preprocess_progress.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        
        # Segmentation progress
        ttk.Label(main_frame, text="Segmentation").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        #self.segment_text = ttk.Label(main_frame, text="0%")
        #self.segment_text.grid(row=4, column=1, sticky=tk.E, pady=(0, 5))
        
        self.segment_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.segment_progress.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        ttk.Label(main_frame, text="Stitching").grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        self.stitching_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.stitching_progress.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        

        # Analysis progress
        ttk.Label(main_frame, text="Analysis").grid(row=8, column=0, sticky=tk.W, pady=(0, 5))
        #self.analysis_text = ttk.Label(main_frame, text="0%")
        #self.analysis_text.grid(row=6, column=1, sticky=tk.E, pady=(0, 5))
        
        self.analysis_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.analysis_progress.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        
        # Status message
        #self.status_message = ttk.Label(main_frame, text="Starting processing...")
        #self.status_message.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 15))
        
        # Cancel button
        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.confirm_cancel)
        self.cancel_button.grid(row=9, column=0, columnspan=2, pady=(0, 10))

    def update_image_progress(self, step, total_steps):
        self.image_progress["value"] = step
        self.image_progress_text.config(text=f"Image {step}/{total_steps}")

        self.reset_process_progress()
        self.root.update_idletasks()

    def reset_process_progress(self):
        """Reset all process progress bars for a new image"""
        self.preprocess_progress["value"] = 0
        #self.preprocess_text.config(text="0%")
        self.segment_progress["value"] = 0  
        #self.segment_text.config(text="0%")
        self.stitching_progress["value"] = 0
        self.analysis_progress["value"] = 0
        #self.analysis_text.config(text="0%")
        
    def update_process_progress(self, process_name, percent):
        percent = min(100, max(0, percent))

        now = datetime.now().strftime('%H:%M:%S')
        
        if process_name.lower() == "preprocessing":
            self.preprocess_progress["value"] = percent
        elif process_name.lower() == "segmentation":
            self.segment_progress["value"] = percent
        elif process_name.lower() == "stitching":
            self.stitching_progress["value"] = percent
        elif process_name.lower() == "analysis":
            self.analysis_progress["value"] = percent

        self.root.update_idletasks()


    def confirm_cancel(self):
        """Show confirmation dialog before cancelling"""
        if messagebox.askyesno("Cancel Processing", 
                            "Are you sure you want to cancel?\nOnly fully processed images will be saved."):
            self.cancelled = True
            #self.status_message.config(text="Cancelling... Please wait.")
            self.cancel_button.config(state=tk.DISABLED)
            self.root.update_idletasks()
            self.close()  # <- Add this to actually close the window
            
    def is_cancelled(self):
        """Check if processing was cancelled"""
        return self.cancelled
    
    def on_closing(self):
        """Handle window close button"""
        self.confirm_cancel()
        
    def close(self):
        """Close the dialog"""
        self.root.grab_release()
        self.root.destroy()

def update_image_progress_dialog(step, total_steps):
    global root_window

    if progress_dialog:
        # Update the progress bar for images
        progress_dialog.update_image_progress(step + 1, total_steps)
        root_window.update()  # Allow UI to update    

def update_progress_dialog(step, total_steps, process):
    global root_window

    if progress_dialog:
        # Update the progress bar for segmentation
        percent = ((step + 1) * 100) // total_steps
        progress_dialog.update_process_progress(process, percent)
        root_window.update()  # Allow UI to update

def initialise_window():
    """Initialise the main application window and UI components."""
    global root_window

    # Default settings
    segmentation_selected = True

    # Create main window
    root_window = tk.Tk()
    
    # Create analysis options window
    analysis_window = tk.Toplevel(root_window)
    analysis_window.title("Run Analysis")
    analysis_window.geometry("400x300")

    # Main frame
    frame = tk.Frame(analysis_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    tk.Label(frame, text="Run Analysis", font=("Arial", 12, "bold")).pack(pady=10)

    # Process selection section
    processes_frame = tk.LabelFrame(frame, text="Select Processes")
    processes_frame.pack(fill=tk.X, pady=10)

    # Process checkboxes
    processes = {
        "Preprocessing": tk.BooleanVar(value=True),
        "Segmentation": tk.BooleanVar(value=segmentation_selected),
        "Analysis": tk.BooleanVar(value=True)
    }

    for process, var in processes.items():
        tk.Checkbutton(processes_frame, text=process, variable=var).pack(anchor="w", padx=10)

    # Parameter display section
    parameters_frame = tk.LabelFrame(frame, text="Parameters")
    parameters_frame.pack(fill=tk.X, pady=10)

    # Buttons section
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=20)

    # Cancel and Start buttons
    tk.Button(
        button_frame, 
        text="Cancel", 
        command=analysis_window.destroy, 
        width=10
    ).pack(side=tk.LEFT, padx=5)
    
    tk.Button(
        button_frame, 
        text="Start", 
        command=lambda: start_analysis(processes), 
        width=10
    ).pack(side=tk.LEFT, padx=5)
    
    # Start main event loop
    root_window.mainloop()

    return processes

def load_project_data(project_path, processes):
    # Check if segmentation is selected from processes.csv
    processes_path = os.path.join(project_path, 'processes.csv')
    if os.path.exists(processes_path):
        processes_df = pd.read_csv(processes_path)
        segmentation_row = processes_df[processes_df['process'] == 'cellpose_nuclear_segmentation']
        segmentation_selected = len(segmentation_row) > 0 and segmentation_row['selected'].values[0] == 'yes'
    else:
        segmentation_selected = True  # Default to True if file doesn't exist

            
    # Define available processes (instead of loading from file)
    available_processes = ['CLAHE', 'Gaussian']

    # Load parameters from parameters_updated.csv
    parameters_path = os.path.join(project_path, 'parameters.csv')
    if os.path.exists(parameters_path):
        parameters_df = pd.read_csv(parameters_path)
        
        print(parameters_df)
        # Load parameters file
        try:
            required_columns = ['parameter', 'process', 'channel', 'value', 'default_value', 'data_type', 'must_be_odd']
            if not all(col in parameters_df.columns for col in required_columns):
                raise ValueError(f"Parameters CSV must contain columns: {required_columns}")
        except Exception as e:
            raise ValueError(f"Error loading parameters CSV file: {str(e)}")
        
        channel_parameters = {}
        global_parameters = {}
        # Dictionary to store process selection state
        process_enabled = {process: False for process in available_processes}
        
        # Initialize parameter values from the CSV
        for _, row in parameters_df.iterrows():
            parameter_name = row['parameter']
            channel = row['channel']
            value = row['value']
            default_value = row['default_value']
            
            # Use value if not NaN, otherwise use default_value
            parameter_value = value if pd.notnull(value) else default_value
            
            if pd.notnull(channel):
                try:
                    channel = int(channel)
                    # Initialize channel dictionary if it doesn't exist
                    if channel not in channel_parameters:
                        channel_parameters[channel] = {}
                    channel_parameters[channel][parameter_name] = parameter_value
                except ValueError:
                    # Not a number, treat as global
                    global_parameters[parameter_name] = parameter_value
            else:
                global_parameters[parameter_name] = parameter_value
        
        # Map string type names from CSV to actual Python types
        dtype_map = {
            'Integer': int,
            'Float': float,
        }

        for _, row in parameters_df.iterrows():
            param_name = row['parameter']
            dtype_str = row['data_type']
            
            
            cast = dtype_map.get(dtype_str)
            if cast is None:
                raise ValueError(f"Unsupported Data Type '{dtype_str}' for parameter '{param_name}'")

        print("Channel parameters:", channel_parameters) 
        print("Global parameters:", global_parameters)


    def find_segmentation_channel():
        channel_details_path = os.path.join(project_path, 'channel_details.csv')
        try:
            df = pd.read_csv(channel_details_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find file: {channel_details_path}")
        
        if 'channel' not in df.columns or 'segmentation_channel' not in df.columns:
            raise ValueError("CSV must have 'channel' and 'segmentation_channel' columns")
        
        segmentation_rows = df[df['segmentation_channel'] == 'yes']
        if len(segmentation_rows) != 1:
            raise ValueError("Must have exactly one segmentation channel marked as 'yes'")
        
        return segmentation_rows.index[0]

    segmentation_channel_index = find_segmentation_channel()
    print("segmentation channel is", segmentation_channel_index)

    # Check if project folder has TIFF files
    tiff_files = [f for f in os.listdir(project_path) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]

    if not tiff_files:
        messagebox.showerror("Error", "No TIFF files found in the project folder.")
        return

    total_files_to_analyse = len(tiff_files)

    # Get selected processes
    selected_processes = [p for p, v in processes.items() if v.get()]
    if not selected_processes:
        messagebox.showerror("Error", "Please select at least one process.")
        return

    return selected_processes, channel_parameters, global_parameters, segmentation_channel_index, tiff_files, total_files_to_analyse

def load_image(image_path):
    # Load the TIFF file
    image = tifffile.imread(image_path)
    print(f"Loaded image with shape: {image.shape}")

    return image

def preprocess(channel_slice, this_channel_parameters):

    image = channel_slice.copy()

    clahe_kernel_size = int(this_channel_parameters['CLAHE_kernel_size'])
    clahe_clip_limit = float(this_channel_parameters['CLAHE_clip_limit'])
    clahe_n_bins = int(this_channel_parameters['CLAHE_n_bins'])
    
    gaussian_sigma = float(this_channel_parameters['gaussian_sigma'])
    gaussian_kernel_size = float(this_channel_parameters['gaussian_kernel_size'])

    def apply_gaussian(image, gaussian_sigma, gaussian_kernel_size):
        """Apply Gaussian blur and CLAHE preprocessing to image"""
        # Apply Gaussian blur
        print("Gaussian", image.dtype)

        gaussian_blurred_image = img_as_ubyte(gaussian(
            image,
            sigma=gaussian_sigma,
            truncate=gaussian_kernel_size))
        
        return gaussian_blurred_image

    def apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        print("CLAHE", image.dtype)

        CLAHE_image = img_as_ubyte(equalize_adapthist(
            image,
            kernel_size=clahe_kernel_size,
            clip_limit=clahe_clip_limit,
            nbins=clahe_n_bins
        ))
        

        print("finished CLAHE", CLAHE_image.dtype)

        return CLAHE_image

    total_preprocessing_steps = 2

    preprocessing_step = 0
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing")
    image = apply_gaussian(image, gaussian_sigma, gaussian_kernel_size)

    preprocessing_step = 1
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing")
    image = apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins)

    return image

def segment_and_stitch(channel_slice, segmentation_parameters):

    cell_diameter = segmentation_parameters['cell_diameter']
    flow_threshold = segmentation_parameters['flow_threshold']
    cellprob_threshold = segmentation_parameters['cellprob_threshold']
    iou_threshold = 0.5


    print("Starting segmentation")
    print(f"Cell diameter = {cell_diameter}, Flow threshold = {flow_threshold}, Cellprob threshold = {cellprob_threshold}")
    print("IoU threshold for stitching = ", iou_threshold)


    def segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold):
        """Segment nuclei in 2D slices using Cellpose"""
        model = models.Cellpose(gpu=False, model_type='nuclei')
        
        total_z = channel_slice.shape[0]
        segmented_image = np.zeros_like(channel_slice, dtype=np.uint16)
        total_cells_segmented = 0
        
        for z in range(total_z):
            update_progress_dialog(z, total_z, "Segmentation")

            z_slice = channel_slice[z]
            segmented_image_z_slice, flows, styles, diams = model.eval(
                z_slice,
                diameter=cell_diameter,
                flow_threshold=flow_threshold,  # Higher is more cells
                cellprob_threshold=cellprob_threshold,  # Lower is more cells
            )
        
            segmented_image[z] = segmented_image_z_slice
            total_cells_segmented += len(np.unique(segmented_image_z_slice)) - 1  # Subtract 1 for background (0)
        
        print(f"Total cells segmented: {total_cells_segmented}")
                      
        return segmented_image

    def stitch_by_iou(segmented_image):
        def calculate_iou(cell_1, cell_2):
            """Calculate Intersection over Union (IoU) between two binary masks"""
            intersection = np.logical_and(cell_1, cell_2).sum()
            union = np.logical_or(cell_1, cell_2).sum()
            if union == 0:
                return 0
            return intersection / union
        
        """Stitch segmented cells across z-slices based on IoU"""
        total_z = segmented_image.shape[0]
        
        def relabel_stitched_masks(segmented_image):
            """Relabels the stitched 2D segmentation masks based on IoU across z-slices"""
            stitched_image = np.squeeze(segmented_image).astype(np.uint16)
                
            current_label = 1
            for z in range(1, total_z):
                #if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
                 #   print(f"{datetime.now():%H:%M:%S} - Stitching {((z + 1) * 100) // total_z}% complete)")
                update_progress_dialog(z-1, total_z-1, "Stitching")

                previous_slice = stitched_image[z-1]
                current_slice = stitched_image[z]
                
                # Create a copy of the current slice to store new labels
                new_labels = np.zeros_like(current_slice)
                
                # Find the unique labels in the current slice
                unique_labels = np.unique(current_slice)
                
                for label in unique_labels:
                    if label == 0:
                        continue  # Skip background

                    # Extract the current cell in the current slice
                    current_cell = current_slice == label
                    
                    # Check for overlap with any cell in the previous slice
                    max_iou = 0
                    best_match_label = 0
                    overlap_labels = np.unique(previous_slice[current_cell])
                    overlap_labels = overlap_labels[overlap_labels > 0]  # Exclude background
                    
                    for previous_label in overlap_labels:
                        previous_cell = previous_slice == previous_label
                        iou = calculate_iou(current_cell, previous_cell)
                        if iou > max_iou:
                            max_iou = iou
                            best_match_label = previous_label
                    
                    if max_iou >= iou_threshold:
                        # If the IoU is above the threshold, assign the previous label
                        new_labels[current_cell] = best_match_label
                    else:
                        # Otherwise, assign a new label
                        new_labels[current_cell] = current_label
                        current_label += 1
                
                # Update the current slice with the new labels
                stitched_image[z] = new_labels

            return stitched_image

        relabelled_stitched_masks = relabel_stitched_masks(segmented_image)

        return relabelled_stitched_masks

    def clean_labels(stitched_image):
        """Relabels segmented image so that every cell has a unique label and none are skipped"""
        # Get the unique values in the segmented image, excluding 0 (background)
        unique_labels = np.unique(stitched_image)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
        # Create a mapping from old labels to new labels (keep 0 as background)
        label_mapping = {0: 0}  # Keep background as 0
        for new_label, old_label in enumerate(unique_labels, start=1):
            label_mapping[old_label] = new_label
        
        # Relabel the segmented image using the mapping
        relabeled_image = np.zeros_like(stitched_image)
        for old_label, new_label in label_mapping.items():
            relabeled_image[stitched_image == old_label] = new_label
        
        return relabeled_image

    segmented_image = segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold)
    stitched_image = stitch_by_iou(segmented_image)
    cleaned_segmented_image = clean_labels(stitched_image)

    return segmented_image, cleaned_segmented_image

def process_single_image(image, segmentation_channel_index, selected_processes, channel_parameters, global_parameters, 
                        output_folder, tiff_file):
    """Process a single image through the selected processing steps."""
    preprocessed_image = image.copy()
    segmented_image = None
    file_base_name = os.path.splitext(tiff_file)[0]

    preprocessed_path = os.path.join(output_folder, f"{file_base_name}_preprocessed.tiff")


    # PREPROCESSING
    if "Preprocessing" in selected_processes and not progress_dialog.is_cancelled():
        #print(f"{datetime.now():%H:%M:%S} - Starting preprocessing")
        
        # Run preprocessing

        total_channels = image.shape[1] if len(image.shape) > 3 else 1

        for channel in range(total_channels):
            this_channel_parameters = channel_parameters[channel]
            print("image shape", image.shape)
            channel_slice = image[:, channel, :, :]
            print("channel_slice_shape", channel_slice.shape)

            preprocessed_image[:, channel, : :] = preprocess(channel_slice, this_channel_parameters)
        
        # Save preprocessed image
        tifffile.imwrite(preprocessed_path, preprocessed_image)
        #print(f"Saved preprocessed image to {preprocessed_path}")
    else:
        # Check if preprocessed file already exists and load it
        if os.path.exists(preprocessed_path):
            preprocessed_image = tifffile.imread(preprocessed_path)
            print(f"Loaded existing preprocessed image from {preprocessed_path}") 
            print("Shape", preprocessed_image.shape)       

    # SEGMENTATION
    if "Segmentation" in selected_processes and not progress_dialog.is_cancelled():
        #print(f"{datetime.now():%H:%M:%S} - Starting segmentation")
        
        # Use preprocessed image if available, otherwise use raw segmentation channel

        if preprocessed_image is None:
            segmentation_input = image[:, segmentation_channel_index, :, :]
        else:
            segmentation_input = preprocessed_image[:, segmentation_channel_index, :, :]

        segmentation_parameters = global_parameters

        # Run segmentation
        raw_segmented_image, segmented_image = segment_and_stitch(segmentation_input, segmentation_parameters)
        
        # Save segmentation result if not cancelled
        if not progress_dialog.is_cancelled() and segmented_image is not None:
            raw_segmentation_path = os.path.join(output_folder, f"{file_base_name}_raw_segmentation.tiff")
            segmentation_path = os.path.join(output_folder, f"{file_base_name}_segmentation.tiff")
            tifffile.imwrite(segmentation_path, segmented_image)
            tifffile.imwrite(raw_segmentation_path, raw_segmented_image)

    # ANALYSIS
    if "Analysis" in selected_processes and not progress_dialog.is_cancelled():
        print(f"{datetime.now():%H:%M:%S} - Starting analysis")
        
        # Run analysis (replace with actual analysis code)
        #run_analysis(image, segmented_image, parameters, output_folder, file_base_name, progress_dialog)
        #print("Running analysis")

    return preprocessed_image, segmented_image

def display(image, preprocessed_image, segmented_image):
    if image is not None and display == True:
        viewer = napari.Viewer()
        viewer.add_image(image, name='Original')
        
        if preprocessed_image is not None:
            viewer.add_image(preprocessed_image, name='Preprocessed')
        
        if segmented_image is not None:
            viewer.add_labels(segmented_image, name='Segmentation')
        
        napari.run()

def process_all_images(project_path, selected_processes, channel_parameters, global_parameters, segmentation_channel_index, 
                     tiff_files, total_files):
    """Process all image files in the project."""
    
    for file_index, tiff_file in enumerate(tiff_files):
        # Update progress indicator
        update_image_progress_dialog(
            step=file_index, 
            total_steps=total_files
        )

        # Set up paths
        tiff_path = os.path.join(project_path, tiff_file)
        output_folder = os.path.join(project_path, os.path.splitext(tiff_file)[0])
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Load image data
        image = load_image(tiff_path)
        
        process_single_image(
                image,
                segmentation_channel_index,
                selected_processes, 
                channel_parameters,
                global_parameters,
                output_folder, 
                tiff_file
            )

def start_analysis(processes):
    global progress_dialog

    """Initialise and start the analysis process."""
    # Project configuration
    project_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis")

    # Load required data for processing
    selected_processes, channel_parameters, global_parameters, segmentation_channel_index, tiff_files, total_files = load_project_data(
        project_path, 
        processes
    )

    # Initialise progress dialog
    progress_dialog = progressdialog(
        parent=root_window,
        title="Processing Images", 
        total_images=total_files,
    )

    # Start the analysis pipeline as a thread, so GUI Cancel button is responsive
    processing_thread = threading.Thread(
    target=process_all_images,
    args=
        (project_path, 
        selected_processes, 
        channel_parameters,
        global_parameters,        
        segmentation_channel_index, 
        tiff_files, 
        total_files)
    )
    processing_thread.daemon = True  # So it doesn't block shutdown if needed
    processing_thread.start()   

#%%
#RUN ALL
processes = initialise_window()

#%%
#OBTAIN DATA NEEDED TO RUN DEBUGGING CELL
project_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis")

parameters_path = os.path.join(project_path, 'parameters.csv')
if os.path.exists(parameters_path):
    parameters_df = pd.read_csv(parameters_path)
    
    # Extract parameters with correct data type
    parameters = {}
    for _, row in parameters_df.iterrows():
        parameter_name = row['parameter']
        parameter_value = row['value']
        data_type = row['data_type']
        
        if data_type == 'Float':
            parameters[parameter_name] = float(parameter_value)
        elif data_type == 'Integer':
            parameters[parameter_name] = int(parameter_value)
        else:
            parameters[parameter_name] = parameter_value

image_path = os.path.normpath(r"Y:\Room225_SharedFolder\Leica_Stellaris5_data\Gastruloids\oskar\analysis\SBSO_OPP_NM_two_analysis\replicate_1.tif")

image = tifffile.imread(image_path)

#%%
#RUN ONLY DEBUGGING CELL
segment_and_stitch(image, parameters)