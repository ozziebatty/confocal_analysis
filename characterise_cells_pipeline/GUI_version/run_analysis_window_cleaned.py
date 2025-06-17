print("Importing packages...")
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
from skimage import img_as_ubyte
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

def load_project_data(project_root, processes):
    # Check if segmentation is selected from processes.csv
    processes_path = os.path.join(project_root, 'processes.csv')
    if os.path.exists(processes_path):
        processes_df = pd.read_csv(processes_path)
        segmentation_row = processes_df[processes_df['Process'] == 'Cellpose nuclear segmentation']
        segmentation_selected = len(segmentation_row) > 0 and segmentation_row['Selected'].values[0] == 'Yes'
    else:
        segmentation_selected = True  # Default to True if file doesn't exist

    # Load parameters from parameters_updated.csv
    parameters_path = os.path.join(project_root, 'parameters_updated.csv')
    if os.path.exists(parameters_path):
        parameters_df = pd.read_csv(parameters_path)
        
        # Extract parameters with correct data type
        parameters = {}
        for _, row in parameters_df.iterrows():
            parameter_name = row['Parameter']
            parameter_value = row['Value']
            data_type = row['Data type']
            
            if data_type == 'Float':
                parameters[parameter_name] = float(parameter_value)
            elif data_type == 'Integer':
                parameters[parameter_name] = int(parameter_value)
            else:
                parameters[parameter_name] = parameter_value

    # Determine nuclear channel from channel_details.csv
    channel_path = os.path.join(project_root, 'channel_details.csv')
    nuclear_channel_index = 0  # Default to first channel

    if os.path.exists(channel_path):
        channel_df = pd.read_csv(channel_path)
        nuclear_rows = channel_df[channel_df['Nuclear Channel'] == 'Yes']
        
        if len(nuclear_rows) > 0:
            # Get the index from the row index (B2 -> index 0, B6 -> index 4)
            # Assuming B2, B3, B4, etc. correspond to rows 1, 2, 3, etc.
            nuclear_channel_index = nuclear_rows.index[0]

    # Check if project folder has TIFF files
    tiff_files = [f for f in os.listdir(project_root) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]

    if not tiff_files:
        messagebox.showerror("Error", "No TIFF files found in the project folder.")
        return

    total_files_to_analyse = len(tiff_files)

    # Get selected processes
    selected_processes = [p for p, v in processes.items() if v.get()]
    if not selected_processes:
        messagebox.showerror("Error", "Please select at least one process.")
        return

    return selected_processes, parameters, nuclear_channel_index, tiff_files, total_files_to_analyse

def load_image_data(tiff_path, nuclear_channel_index):
    # Load the TIFF file
    image = tifffile.imread(tiff_path)
    print(f"Loaded image with shape: {image.shape}")
    
    # Extract nuclear channel
    if len(image.shape) > 3:  # If multi-channel
        nuclear_slice = image[:, nuclear_channel_index, :, :]
    else:  # Single channel
        nuclear_slice = image

    return image, nuclear_slice

def preprocess(nuclear_slice, parameters):

    image = nuclear_slice

    clahe_kernel_size = parameters.get('CLAHE_kernel_size', 64)
    clahe_clip_limit = parameters.get('CLAHE_cliplimit', 0.01)
    clahe_n_bins = parameters.get('CLAHE_n_bins', 256)
    gaussian_kernel_size = parameters.get('gaussian_kernel_size', 3)
    gaussian_sigma = parameters.get('gaussian_sigma', 1.0)

    def apply_gaussian(image, gaussian_sigma, gaussian_kernel_size):
        """Apply Gaussian blur and CLAHE preprocessing to image"""
        # Apply Gaussian blur
        gaussian_blurred_image = gaussian(image, sigma=gaussian_sigma, truncate=gaussian_kernel_size)
        
        return gaussian_blurred_image

    def apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        CLAHE_image = img_as_ubyte(equalize_adapthist(
            image, 
            kernel_size=clahe_kernel_size, 
            clip_limit=clahe_clip_limit, 
            nbins=clahe_n_bins
        ))
        
        return CLAHE_image

    total_preprocessing_steps = 2

    preprocessing_step = 0
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing")
    image = apply_gaussian(image, gaussian_sigma, gaussian_kernel_size)

    preprocessing_step = 1
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing")
    image = apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins)

    return image

def segment_and_stitch(image, parameters):

    print("working")

    cell_diameter = parameters.get('cell_diameter', 30)
    flow_threshold = parameters.get('flow_threshold', 0.5)
    cellprob_threshold = parameters.get('cellprob_threshold', 0.1)
    iou_threshold = parameters.get('iou_threshold', 0.5)

    def segment_2D(channel_slice):
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
        #print(f"Cell diameter = {cell_diameter}, Flow threshold = {flow_threshold}")
                      
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

    segmented_image = segment_2D(image)
    stitched_image = stitch_by_iou(image)
    cleaned_segmented_image = clean_labels(stitched_image)

    return cleaned_segmented_image

def process_single_image(image, nuclear_slice, selected_processes, parameters, 
                        output_folder, tiff_file):
    """Process a single image through the selected processing steps."""
    preprocessed_image = None
    segmented_image = None
    file_base_name = os.path.splitext(tiff_file)[0]

    # PREPROCESSING
    if "Preprocessing" in selected_processes and not progress_dialog.is_cancelled():
        #print(f"{datetime.now():%H:%M:%S} - Starting preprocessing")
        
        # Run preprocessing
        preprocessed_image = preprocess(nuclear_slice, parameters)
        
        # Save preprocessed image
        preprocessed_path = os.path.join(output_folder, f"{file_base_name}_preprocessed.tiff")
        tifffile.imwrite(preprocessed_path, preprocessed_image)
        #print(f"Saved preprocessed image to {preprocessed_path}")

    # SEGMENTATION
    if "Segmentation" in selected_processes and not progress_dialog.is_cancelled():
        #print(f"{datetime.now():%H:%M:%S} - Starting segmentation")
        
        # Use preprocessed image if available, otherwise use raw nuclear channel
        segmentation_input = preprocessed_image if preprocessed_image is not None else nuclear_slice
        
        # Run segmentation
        segmented_image = segment_and_stitch(segmentation_input, parameters)
        
        # Save segmentation result if not cancelled
        if not progress_dialog.is_cancelled() and segmented_image is not None:
            segmentation_path = os.path.join(output_folder, f"{file_base_name}_segmentation.tiff")
            tifffile.imwrite(segmentation_path, segmented_image)
            #print(f"Saved segmentation to {segmentation_path}")

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

def process_all_images(project_root, selected_processes, parameters, nuclear_channel_index, 
                     tiff_files, total_files):
    """Process all image files in the project."""
    
    for file_index, tiff_file in enumerate(tiff_files):
        # Update progress indicator
        update_image_progress_dialog(
            step=file_index, 
            total_steps=total_files
        )

        # Set up paths
        tiff_path = os.path.join(project_root, tiff_file)
        output_folder = os.path.join(project_root, os.path.splitext(tiff_file)[0])
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Load image data
        image, nuclear_slice = load_image_data(tiff_path, nuclear_channel_index)
        
        process_single_image(
                image, 
                nuclear_slice, 
                selected_processes, 
                parameters, 
                output_folder, 
                tiff_file
            )

def start_analysis(processes):
    global progress_dialog

    """Initialise and start the analysis process."""
    # Project configuration
    project_root = "/Users/oskar/Desktop/format_test"

    # Load required data for processing
    selected_processes, parameters, nuclear_channel_index, tiff_files, total_files = load_project_data(
        project_root, 
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
        (project_root, 
        selected_processes, 
        parameters, 
        nuclear_channel_index, 
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
project_root = "/Users/oskar/Desktop/format_test"

parameters_path = os.path.join(project_root, 'parameters_updated.csv')
if os.path.exists(parameters_path):
    parameters_df = pd.read_csv(parameters_path)
    
    # Extract parameters with correct data type
    parameters = {}
    for _, row in parameters_df.iterrows():
        parameter_name = row['Parameter']
        parameter_value = row['Value']
        data_type = row['Data type']
        
        if data_type == 'Float':
            parameters[parameter_name] = float(parameter_value)
        elif data_type == 'Integer':
            parameters[parameter_name] = int(parameter_value)
        else:
            parameters[parameter_name] = parameter_value

image = tifffile.imread('/Users/oskar/Desktop/format_test/SBSO_stellaris_cropped.tiff')

#%%
#RUN ONLY DEBUGGING CELL
segment_and_stitch(image, parameters)