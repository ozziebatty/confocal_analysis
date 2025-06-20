print("IMPORTING")
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
import tifffile
import numpy as np
import pandas as pd
import threading
from datetime import datetime

# Import the analysis modules at the top
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from cellpose import models

# Global variables
loaded_project = True
project_path = None
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
        self.preprocess_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.preprocess_progress.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        
        # Segmentation progress
        ttk.Label(main_frame, text="Segmentation").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.segment_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.segment_progress.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        ttk.Label(main_frame, text="Stitching").grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        self.stitching_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.stitching_progress.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        # Analysis progress
        ttk.Label(main_frame, text="Analysis").grid(row=8, column=0, sticky=tk.W, pady=(0, 5))
        self.analysis_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
        self.analysis_progress.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
        
        # Cancel button
        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.confirm_cancel)
        self.cancel_button.grid(row=10, column=0, columnspan=2, pady=(0, 10))

    def update_image_progress(self, step, total_steps):
        self.image_progress["value"] = step
        self.image_progress_text.config(text=f"Image {step}/{total_steps}")
        self.reset_process_progress()
        self.root.update_idletasks()

    def reset_process_progress(self):
        """Reset all process progress bars for a new image"""
        self.preprocess_progress["value"] = 0
        self.segment_progress["value"] = 0  
        self.stitching_progress["value"] = 0
        self.analysis_progress["value"] = 0
        
    def update_process_progress(self, process_name, percent):
        percent = min(100, max(0, percent))
        
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
            self.cancel_button.config(state=tk.DISABLED)
            self.root.update_idletasks()
            self.close()
            
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

def update_image_progress_dialog(step, total_steps, root_window, progress_dialog):

    if progress_dialog:
        progress_dialog.update_image_progress(step + 1, total_steps)
        root_window.update()

def update_progress_dialog(step, total_steps, process, root_window, progress_dialog):
    if progress_dialog:
        percent = ((step + 1) * 100) // total_steps
        progress_dialog.update_process_progress(process, percent)
        root_window.update()

def load_project_data(project_path, processes):
    # Check if segmentation is selected from processes.csv
    processes_path = os.path.join(project_path, 'processes.csv')
    if os.path.exists(processes_path):
        processes_df = pd.read_csv(processes_path)
        segmentation_row = processes_df[processes_df['process'] == 'cellpose_nuclear_segmentation']
        segmentation_selected = len(segmentation_row) > 0 and segmentation_row['selected'].values[0] == 'yes'
    else:
        segmentation_selected = True  # Default to True if file doesn't exist

    # Define available processes
    available_processes = ['CLAHE', 'Gaussian']

    # Load parameters from parameters.csv
    parameters_path = os.path.join(project_path, 'parameters.csv')
    if os.path.exists(parameters_path):
        parameters_df = pd.read_csv(parameters_path)
        
        # Load parameters file
        try:
            required_columns = ['parameter', 'process', 'channel', 'value', 'default_value', 'data_type', 'must_be_odd']
            if not all(col in parameters_df.columns for col in required_columns):
                raise ValueError(f"Parameters CSV must contain columns: {required_columns}")
        except Exception as e:
            raise ValueError(f"Error loading parameters CSV file: {str(e)}")
        
        channel_parameters = {}
        global_parameters = {}
        
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
                    if channel not in channel_parameters:
                        channel_parameters[channel] = {}
                    channel_parameters[channel][parameter_name] = parameter_value
                except ValueError:
                    global_parameters[parameter_name] = parameter_value
            else:
                global_parameters[parameter_name] = parameter_value

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

def preprocess(channel_slice, this_channel_parameters, root_window, progress_dialog):
    image = channel_slice.copy()

    clahe_kernel_size = int(this_channel_parameters['CLAHE_kernel_size'])
    clahe_clip_limit = float(this_channel_parameters['CLAHE_clip_limit'])
    clahe_n_bins = int(this_channel_parameters['CLAHE_n_bins'])
    
    gaussian_sigma = float(this_channel_parameters['gaussian_sigma'])
    gaussian_kernel_size = float(this_channel_parameters['gaussian_kernel_size'])

    def apply_gaussian(image, gaussian_sigma, gaussian_kernel_size):
        print("applying gaussian")
        gaussian_blurred_image = img_as_ubyte(gaussian(
            image,
            sigma=gaussian_sigma,
            truncate=gaussian_kernel_size))
        return gaussian_blurred_image

    def apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins):
        print("applying CLAHE")
        CLAHE_image = img_as_ubyte(equalize_adapthist(
            image,
            kernel_size=clahe_kernel_size,
            clip_limit=clahe_clip_limit,
            nbins=clahe_n_bins
        ))
        return CLAHE_image

    total_preprocessing_steps = 2

    preprocessing_step = 0
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing", root_window=root_window, progress_dialog=progress_dialog)
    image = apply_gaussian(image, gaussian_sigma, gaussian_kernel_size)

    preprocessing_step = 1
    update_progress_dialog(step=preprocessing_step, total_steps=total_preprocessing_steps, process="preprocessing", root_window=root_window, progress_dialog=progress_dialog)
    image = apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins)

    return image

def segment_and_stitch(channel_slice, segmentation_parameters, root_window, progress_dialog):
    cell_diameter = segmentation_parameters['cell_diameter']
    flow_threshold = segmentation_parameters['flow_threshold']
    cellprob_threshold = segmentation_parameters['cellprob_threshold']
    iou_threshold = 0.5

    def segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold):
        """Segment nuclei in 2D slices using Cellpose"""
        model = models.Cellpose(gpu=False, model_type='nuclei')
        
        total_z = channel_slice.shape[0]
        segmented_image = np.zeros_like(channel_slice, dtype=np.uint16)
        total_cells_segmented = 0
        
        for z in range(total_z):
            update_progress_dialog(z, total_z, "Segmentation", root_window, progress_dialog)
            print(z, total_z, "segmentation")

            z_slice = channel_slice[z]
            segmented_image_z_slice, flows, styles, diams = model.eval(
                z_slice,
                diameter=cell_diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
        
            segmented_image[z] = segmented_image_z_slice
            total_cells_segmented += len(np.unique(segmented_image_z_slice)) - 1
        
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
        
        total_z = segmented_image.shape[0]
        
        def relabel_stitched_masks(segmented_image):
            """Relabels the stitched 2D segmentation masks based on IoU across z-slices"""
            stitched_image = np.squeeze(segmented_image).astype(np.uint16)
                
            current_label = 1
            for z in range(1, total_z):
                update_progress_dialog(z-1, total_z-1, "Stitching")

                previous_slice = stitched_image[z-1]
                current_slice = stitched_image[z]
                
                new_labels = np.zeros_like(current_slice)
                unique_labels = np.unique(current_slice)
                
                for label in unique_labels:
                    if label == 0:
                        continue

                    current_cell = current_slice == label
                    
                    max_iou = 0
                    best_match_label = 0
                    overlap_labels = np.unique(previous_slice[current_cell])
                    overlap_labels = overlap_labels[overlap_labels > 0]
                    
                    for previous_label in overlap_labels:
                        previous_cell = previous_slice == previous_label
                        iou = calculate_iou(current_cell, previous_cell)
                        if iou > max_iou:
                            max_iou = iou
                            best_match_label = previous_label
                    
                    if max_iou >= iou_threshold:
                        new_labels[current_cell] = best_match_label
                    else:
                        new_labels[current_cell] = current_label
                        current_label += 1
                
                stitched_image[z] = new_labels

            return stitched_image

        relabelled_stitched_masks = relabel_stitched_masks(segmented_image)
        return relabelled_stitched_masks

    def clean_labels(stitched_image):
        """Relabels segmented image so that every cell has a unique label and none are skipped"""
        unique_labels = np.unique(stitched_image)
        unique_labels = unique_labels[unique_labels != 0]
        
        label_mapping = {0: 0}
        for new_label, old_label in enumerate(unique_labels, start=1):
            label_mapping[old_label] = new_label
        
        relabeled_image = np.zeros_like(stitched_image)
        for old_label, new_label in label_mapping.items():
            relabeled_image[stitched_image == old_label] = new_label
        
        return relabeled_image

    segmented_image = segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold)
    stitched_image = stitch_by_iou(segmented_image)
    cleaned_segmented_image = clean_labels(stitched_image)

    return segmented_image, cleaned_segmented_image

def process_single_image(image, segmentation_channel_index, selected_processes, channel_parameters, global_parameters, 
                        output_folder, tiff_file, root_window, progress_dialog):
    """Process a single image through the selected processing steps."""
    preprocessed_image = image.copy()
    segmented_image = None
    file_base_name = os.path.splitext(tiff_file)[0]

    preprocessed_path = os.path.join(output_folder, f"{file_base_name}_preprocessed.tiff")

    # PREPROCESSING
    if "Preprocessing" in selected_processes and not progress_dialog.is_cancelled():
        total_channels = image.shape[1] if len(image.shape) > 3 else 1

        for channel in range(total_channels):
            this_channel_parameters = channel_parameters[channel]
            channel_slice = image[:, channel, :, :]
            preprocessed_image[:, channel, :, :] = preprocess(channel_slice, this_channel_parameters, root_window, progress_dialog)
        
        # Save preprocessed image
        tifffile.imwrite(preprocessed_path, preprocessed_image)
    else:
        # Check if preprocessed file already exists and load it
        if os.path.exists(preprocessed_path):
            preprocessed_image = tifffile.imread(preprocessed_path)

    # SEGMENTATION
    if "Segmentation" in selected_processes and not progress_dialog.is_cancelled():
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
        # Add your analysis code here

    return preprocessed_image, segmented_image

def process_all_images(project_path, selected_processes, channel_parameters, global_parameters, segmentation_channel_index, 
                     tiff_files, total_files, root_window, progress_dialog):
    """Process all image files in the project."""
    
    for file_index, tiff_file in enumerate(tiff_files):
        # Check for cancellation
        if progress_dialog.is_cancelled():
            break
            
        # Update progress indicator
        update_image_progress_dialog(
            step=file_index, 
            total_steps=total_files,
            root_window=root_window,
            progress_dialog=progress_dialog
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
                tiff_file,
                root_window=root_window,
                progress_dialog=progress_dialog
            )
    
    # Close progress dialog when done
    if progress_dialog:
        progress_dialog.close()

def create_project_window():
    def create_project():
        input_path = file_entry.get()
        dest_folder = dest_folder_entry.get()
        
        if not input_path or not dest_folder:
            messagebox.showerror("Input Error", "Please provide both input file/folder and destination folder.")
            return

        channel_names = [entry.get().strip() for entry in channel_entries]
        if all(not name for name in channel_names):
            messagebox.showerror("Input Error", "At least one channel name field must be filled.")
            return

        if contains_segmentation_var.get():
            segmentation_index = segmentation_var.get()
            if segmentation_index == -1 or not channel_names[segmentation_index]:
                messagebox.showerror("Input Error", "Segmentation channel must be a valid, non-empty channel.")
                return

        # Create destination if it doesn't exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Copy input file/folder
        try:
            if os.path.isdir(input_path):
                shutil.copytree(input_path, os.path.join(dest_folder, os.path.basename(input_path)))
            else:
                shutil.copy(input_path, dest_folder)
        except Exception as e:
            messagebox.showerror("Copy Error", f"Error copying files: {str(e)}")
            return

        # Save channel details
        try:
            with open(os.path.join(dest_folder, "channel_details.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["channel", "segmentation_channel"])
                for idx, name in enumerate(channel_names):
                    if name:  # Only write non-empty channel names
                        writer.writerow([name, 'yes' if segmentation_var.get() == idx else 'No'])
        except Exception as e:
            messagebox.showerror("File Error", f"Error saving channel details: {str(e)}")
            return

        # Update globals and UI
        global loaded_project, project_path
        loaded_project = True
        project_path = dest_folder
        update_main_page()

        messagebox.showinfo("Success", f"Project created successfully in {dest_folder}")
        project_window.destroy()
    
    project_window = tk.Toplevel(root)
    project_window.title("Create Project")
    project_window.geometry("700x550")
    
    frame = tk.Frame(project_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Input file/folder
    input_frame = tk.Frame(frame)
    input_frame.pack(fill=tk.X, pady=5)
    tk.Label(input_frame, text="Input File/Folder:").pack(side=tk.LEFT)
    file_entry = tk.Entry(input_frame, width=40)
    file_entry.pack(side=tk.LEFT, padx=5)
    
    def browse_input():
        path = filedialog.askdirectory()
        if path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, os.path.normpath(path))

    tk.Button(input_frame, text="Browse", command=browse_input).pack(side=tk.LEFT)
    
    # Destination folder
    dest_frame = tk.Frame(frame)
    dest_frame.pack(fill=tk.X, pady=5)
    tk.Label(dest_frame, text="Destination Folder:").pack(side=tk.LEFT)
    dest_folder_entry = tk.Entry(dest_frame, width=40)
    dest_folder_entry.pack(side=tk.LEFT, padx=5)
    
    def browse_destination():
        project_path = filedialog.askdirectory()
        if project_path:
            dest_folder_entry.delete(0, tk.END)
            dest_folder_entry.insert(0, os.path.normpath(project_path))

    tk.Button(dest_frame, text="Browse", command=browse_destination).pack(side=tk.LEFT)
    
    # segmentation channel option
    segmentation_frame = tk.Frame(frame)
    segmentation_frame.pack(fill=tk.X, pady=5)
    
    contains_segmentation_var = tk.BooleanVar()
    tk.Checkbutton(segmentation_frame, text="Contains Segmentation Channel", variable=contains_segmentation_var).pack(anchor="w")
    
    segmentation_var = tk.IntVar(value=-1)
    
    # Channel entries
    channels_frame = tk.LabelFrame(frame, text="Channel Names")
    channels_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    channel_entries = []
    radio_buttons = []
    
    for i in range(8):
        channel_frame = tk.Frame(channels_frame)
        channel_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(channel_frame, text=f"Channel {i+1}:", width=10).pack(side=tk.LEFT)
        entry = tk.Entry(channel_frame, width=25)
        entry.pack(side=tk.LEFT, padx=5)
        channel_entries.append(entry)
        
        radio = tk.Radiobutton(channel_frame, text="Segmentation", variable=segmentation_var, value=i)
        radio.pack(side=tk.LEFT)
        radio_buttons.append(radio)
        
        # Initially disable all radio buttons
        radio.config(state=tk.DISABLED)
    
    # Function to toggle radio buttons
    def toggle_segmentation_options():
        state = tk.NORMAL if contains_segmentation_var.get() else tk.DISABLED
        for rb in radio_buttons:
            rb.config(state=state)
    
    # Bind checkbox to toggle function
    contains_segmentation_var.trace("w", lambda *args: toggle_segmentation_options())
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Cancel", command=project_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Create Project", command=create_project, width=15).pack(side=tk.LEFT, padx=5)

def define_parameters_window():

    # ====== STEP 1: FILE PATHS AND ERROR CHECKING ======
    if not project_path:
        messagebox.showerror("Error", "Please load a project first.")
        return
    
    print("PROJECT PATH - ", project_path)

    input_image_path = filedialog.askopenfilename(title="Select Sample Image File")
    input_parameters_path = os.path.join(project_path, 'parameters_template.csv')
    output_parameters_path = os.path.join(project_path, 'parameters.csv')

    print("IMPORTING PACKAGES...")
    from skimage.exposure import equalize_adapthist
    from skimage import img_as_ubyte
    import napari
    from magicgui import magicgui
    from cv2 import GaussianBlur
    from magicgui.widgets import Container, CheckBox
    from cellpose import models
    from skimage.segmentation import find_boundaries


    # Store current slice information as global variables for easy access
    current_z = 0
    #current_channel = 0
    changed_channel = False
    
    segmentation_selected = True

    # ====== STEP 2: LOAD DATA ======
    print("INPUT IMAGE PATH - ", input_image_path)
    try:
        # Load image
        print("Loading image...")
        image = tifffile.imread(input_image_path)
        print("Successfully loaded")
        
        # What datatype?
        print(image.min())
        print(image.max())
        
        print("Image loaded with shape (z, c, y, x):", image.shape, "dtype:", image.dtype)
        
        if len(image.shape) != 4:
            raise ValueError(f"Expected 4D image (z, c, y, x), got shape {image.shape}")
        
        total_z = image.shape[0]
        total_channels = image.shape[1]
        min_xy_size = min(image.shape[2], image.shape[3])
        
        # Load parameters file
        try:
            parameters_df = pd.read_csv(input_parameters_path)
            required_columns = ['parameter', 'process', 'channel', 'default_value', 'min', 'max', 'data_type', 'must_be_odd']
            if not all(col in parameters_df.columns for col in required_columns):
                raise ValueError(f"Parameters CSV must contain columns: {required_columns}")
        except Exception as e:
            raise ValueError(f"Error loading parameters CSV file: {str(e)}")
            
        # Define available processes (instead of loading from file)
        available_processes = ['CLAHE', 'Gaussian']
            
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print(image.shape)
        raise

   # Dictionary to store current parameter values
    channel_parameters = {}
    global_parameters = {}

    # Dictionary to store process selection state
    process_enabled = {process: False for process in available_processes}

    # Initialize parameter values from the CSV
    for _, row in parameters_df.iterrows():
        parameter_name = row['parameter']
        channel = row['channel']
        default_value = row['default_value']
        parameter_value = default_value
        # Use default if value is NaN
        
        if pd.notnull(channel):
            try:
                channel = int(channel)
                if channel not in channel_parameters:
                    channel_parameters[channel] = {}
                channel_parameters[channel][parameter_name] = parameter_value
            except ValueError:
                # Not a number, treat as global
                global_parameters[parameter_name] = parameter_value
        else:
            global_parameters[parameter_name] = parameter_value
    
    # Extract min and max values for each parameter from CSV
    # Map string type names from CSV to actual Python types
    dtype_map = {
        'Integer': int,
        'Float': float,
    }

    param_min_max = {}

    for _, row in parameters_df.iterrows():
        param_name = row['parameter']
        dtype_str = row['data_type']
        
        
        cast = dtype_map.get(dtype_str)
        if cast is None:
            raise ValueError(f"Unsupported Data Type '{dtype_str}' for parameter '{param_name}'")

        # Safely cast min/max using the type
        try:
            min_val = cast(row['min'])
            max_val = cast(row['max'])
        except Exception as e:
            raise ValueError(f"Error casting min/max for parameter '{param_name}': {e}")

        param_min_max[param_name] = {'min': min_val, 'max': max_val}
        
    print("Parameter min/max values:", param_min_max)
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

    segmentation_channel = find_segmentation_channel()
    print("segmentation channel is", segmentation_channel)

        
    # ====== STEP 3: DEFINE PROCESSING FUNCTIONS ======
    def apply_CLAHE(image_slice, kernel_size=16, clip_limit=0.005, n_bins=11):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        try:
            # Ensure parameters are valid
            kernel_size = max(3, int(kernel_size))
            clip_limit = max(0.001, float(clip_limit))
            return img_as_ubyte(equalize_adapthist(image_slice, 
                                                kernel_size=(kernel_size, kernel_size), 
                                                clip_limit=clip_limit,
                                                nbins=n_bins))
        except Exception as e:
            print(f"Error applying CLAHE: {str(e)}")
            return image_slice

    def apply_Gaussian(image_slice, kernel_size=11, sigma=0.4):
        """Apply Gaussian blur"""
        try:
            # Ensure kernel size is valid and odd
            kernel_size = max(3, int(kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = max(0.1, float(sigma))
            return GaussianBlur(image_slice, ksize=(kernel_size, kernel_size), sigmaX=sigma)
        except Exception as e:
            print(f"Error applying Gaussian: {str(e)}")
            return image_slice

    def apply_segmentation(image_slice, cell_diameter=8.0, flow_threshold=0.5, cellprob_threshold=0.5):
        """Apply Cellpose segmentation"""
        try:
            print("Applying segmentation with parameters:")
            print(f"- Cell diameter: {cell_diameter}")
            print(f"- Flow threshold: {flow_threshold}")
            print(f"- Cell probability threshold: {cellprob_threshold}")
            
            # Create model - use 'nuclei' for nuclear segmentation
            model = models.Cellpose(gpu=False, model_type='nuclei')
            
            # Run segmentation
            masks, flows, styles, diams = model.eval(
                image_slice,
                diameter=cell_diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
            
            # Create cell outlines (1-pixel thick)
            cell_outlines = find_boundaries(masks, mode='inner', background=0)
            
            print(f"Segmentation complete! Found {np.max(masks)} cells/nuclei")
            return masks, cell_outlines
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty mask with same shape as input
            return np.zeros_like(image_slice), np.zeros_like(image_slice)

    # ====== STEP 4: CREATE NAPARI VIEWER ======
    viewer = napari.Viewer()

    # ====== STEP 5: PREPROCESSING FUNCTION ======
    def apply_preprocessing(current_z, current_channel):
        """Apply selected preprocessing steps to the current slice"""
        global processed_slice, processed_segmentation_slice
        
        try:
            # Get the original slice using global variables
            original_slice = image[current_z, current_channel, :, :]
            processed_slice = original_slice.copy()
            
            # Get selected processes from the global process_enabled dictionary
            selected_processes = [process for process, enabled in process_enabled.items() if enabled]

            # Safely get channel-specific parameters, fallback to empty dict
            this_channel_parameters = channel_parameters.get(current_channel, {})

            # Apply processing steps
            for process in selected_processes:
                if process == 'CLAHE':
                    kernel_size = int(this_channel_parameters.get('CLAHE_kernel_size', 16))
                    clip_limit = float(this_channel_parameters.get('CLAHE_clip_limit', 0.005))
                    n_bins = int(this_channel_parameters.get('CLAHE_n_bins', 11))
                    processed_slice = apply_CLAHE(processed_slice, kernel_size, clip_limit, n_bins)
                    
                elif process == 'Gaussian':
                    kernel_size = int(this_channel_parameters.get('gaussian_kernel_size', 11))
                    sigma = float(this_channel_parameters.get('gaussian_sigma', 0.4))
                    processed_slice = apply_Gaussian(processed_slice, kernel_size, sigma)
        
            # Store the processed image for segmentation
            if current_channel == segmentation_channel:
                processed_segmentation_slice = processed_slice
            
            # Update viewer
            if "Preprocessed" in viewer.layers:
                viewer.layers["Preprocessed"].data = processed_slice
            else:
                viewer.add_image(processed_slice, name="Preprocessed")

            return processed_slice
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            return False
    
    # ====== STEP 6: SLICE SELECTION WIDGET ======
    @magicgui(
        z_slice={"label": "Z slice", "widget_type": "Slider", "min": 0, "max": total_z-1},
        channel={"label": "Channel", "widget_type": "Slider", "min": 0, "max": total_channels-1}, auto_call=True
    )
    def update_slice(z_slice=1, channel=0):
        """Select a z-slice and channel to view"""
        global current_z, current_channel, changed_channel
        
        # Update global variables
        previous_channel = -1
        current_z = z_slice
        current_channel = channel

        if current_channel != previous_channel:
            changed_channel = True
        #print("Current channel when updating:", current_channel)
        
        # Get the original slice
        original_slice = image[z_slice, channel, :, :]
    
        # Update viewer
        if "unprocessed" in viewer.layers:
            viewer.layers["unprocessed"].data = original_slice
        else:
            viewer.add_image(original_slice, name="unprocessed")
            
        # Automatically apply preprocessing after changing slice
        processed_slice = apply_preprocessing(current_z, current_channel)
        
        update_sliders(current_channel)

        return processed_slice

    def refresh_current_slice():
        """Refresh the current slice without changing z or channel"""
        global current_z, current_channel
        
        # Use current global values, with fallbacks
        z_slice = current_z if 'current_z' in globals() else 1
        channel = current_channel if 'current_channel' in globals() else 0
        
        # Get the original slice
        original_slice = image[z_slice, channel, :, :]

        # Update viewer
        if "unprocessed" in viewer.layers:
            viewer.layers["unprocessed"].data = original_slice
        else:
            viewer.add_image(original_slice, name="unprocessed")
        
        # Automatically apply preprocessing after changing slice
        processed_slice = apply_preprocessing(current_z, current_channel)
    
        update_sliders(current_channel)
        return processed_slice

    # ====== STEP 7: CREATE PARAMETER WIDGETS ======
    # Process selection widget
    process_container = Container(layout="vertical")

    # Function to handle process checkbox changes
    def on_process_toggle(process_name, value):
        process_enabled[process_name] = value
        # Apply preprocessing to update the image
        refresh_current_slice()

    # Add checkboxes for each process
    for process in available_processes:
        checkbox = CheckBox(name=f"Enable {process}", value=process_enabled[process])
        # Use lambda with default args to avoid late binding issues
        checkbox.changed.connect(lambda v, p=process: on_process_toggle(p, v))
        process_container.append(checkbox)

    # Combined parameters widget for all processing techniques
    def processing_parameters_function(
        clahe_kernel_size=None,
        clahe_clip_limit=None,
        clahe_n_bins=None,
        gaussian_kernel_size=None,
        gaussian_sigma=None,
    ):
        try:
            # Add a flag to prevent processing during slider updates
            global changed_channel, updating_sliders
            
            # Skip processing if we're currently updating sliders
            if updating_sliders:
                return True

            if changed_channel == True:
                clahe_kernel_size = channel_parameters[current_channel]['CLAHE_kernel_size']
                clahe_clip_limit = channel_parameters[current_channel]['CLAHE_clip_limit']
                clahe_n_bins = channel_parameters[current_channel]['CLAHE_n_bins']
                gaussian_kernel_size = channel_parameters[current_channel]['gaussian_kernel_size']
                gaussian_sigma = channel_parameters[current_channel]['gaussian_sigma']
                changed_channel = False

            # Store updated CLAHE parameters
            else:
                channel_parameters[current_channel]['CLAHE_kernel_size'] = clahe_kernel_size
                channel_parameters[current_channel]['CLAHE_clip_limit'] = clahe_clip_limit
                channel_parameters[current_channel]['CLAHE_n_bins'] = clahe_n_bins
                channel_parameters[current_channel]['gaussian_kernel_size'] = gaussian_kernel_size
                channel_parameters[current_channel]['gaussian_sigma'] = gaussian_sigma

            # Apply preprocessing using updated parameters
            refresh_current_slice()

            return True

        except Exception as e:
            print(f"Error updating parameters: {str(e)}")
            return False


    processing_parameters = magicgui(
            # CLAHE parameters
            clahe_kernel_size={"label": "CLAHE Kernel Size", "widget_type": "Slider", 
                            "min": param_min_max['CLAHE_kernel_size']['min'], "max": min_xy_size, "step": 1},
            clahe_clip_limit={"label": "CLAHE Clip Limit", "widget_type": "FloatSlider", 
                            "min": param_min_max['CLAHE_clip_limit']['min'], "max": param_min_max['CLAHE_clip_limit']['max'], "step": 0.01},
            clahe_n_bins={"label": "CLAHE n bins", "widget_type": "Slider", 
                            "min": param_min_max['CLAHE_n_bins']['min'], "max": param_min_max['CLAHE_n_bins']['max'], "step": 1},

            # Gaussian parameters
            gaussian_kernel_size={"label": "Gaussian Kernel Size", "widget_type": "Slider", 
                                "min": param_min_max['gaussian_kernel_size']['min'], "max": min_xy_size, "step": 2},
            gaussian_sigma={"label": "Gaussian Sigma", "widget_type": "FloatSlider", 
                        "min": param_min_max['gaussian_sigma']['min'], "max": param_min_max['gaussian_sigma']['max'], "step": 0.001},
            auto_call=True
        )(processing_parameters_function)

    def update_sliders(current_channel):
        global updating_sliders
                
        # Set flag to prevent processing during updates
        updating_sliders = True
        
        try:
            processing_parameters['clahe_kernel_size'].value = channel_parameters[current_channel]['CLAHE_kernel_size']
            processing_parameters['clahe_clip_limit'].value = channel_parameters[current_channel]['CLAHE_clip_limit']
            processing_parameters['clahe_n_bins'].value = channel_parameters[current_channel]['CLAHE_n_bins']
            processing_parameters['gaussian_kernel_size'].value = channel_parameters[current_channel]['gaussian_kernel_size']
            processing_parameters['gaussian_sigma'].value = channel_parameters[current_channel]['gaussian_sigma']
        finally:
            # Always reset the flag, even if there's an error
            updating_sliders = False

    # Initialize the flag
    updating_sliders = False

    # ====== STEP 8: SEGMENTATION WIDGET AND FUNCTION ======
    # Segmentation widget without run button
    if segmentation_selected == True:
        # Get min/max values for segmentation parameters
        
        cell_diameter_min = param_min_max['cell_diameter']['min']
        cell_diameter_max = param_min_max['cell_diameter']['max']
        
        flow_threshold_min = param_min_max['flow_threshold']['min']
        flow_threshold_max = param_min_max['flow_threshold']['max']
        
        cell_probability_threshold_min = param_min_max['cellprob_threshold']['min']
        cell_probability_threshold_max = param_min_max['cellprob_threshold']['max']
        
        @magicgui(
            cell_diameter={"label": "Cell diameter", "widget_type": "FloatSlider", 
                        "min": cell_diameter_min, "max": cell_diameter_max, "step": 0.5},
            flow_threshold={"label": "Flow threshold", "widget_type": "FloatSlider", 
                        "min": flow_threshold_min, "max": flow_threshold_max, "step": 0.01},
            cellprob_threshold={"label": "Cell probability threshold", "widget_type": "FloatSlider", 
                            "min": cell_probability_threshold_min, "max": cell_probability_threshold_max, "step": 0.1}
        )
        def segmentation_widget(cell_diameter=8.0, flow_threshold=0.5, cellprob_threshold=0.5):
            """Set parameters for Cellpose segmentation"""
            try:
                # Store parameters
                global_parameters['cell_diameter'] = cell_diameter
                global_parameters['flow_threshold'] = flow_threshold
                global_parameters['cellprob_threshold'] = cellprob_threshold
                
                # Run segmentation with new parameters
                run_segmentation()
                
                return True
            except Exception as e:
                print(f"Error setting segmentation parameters: {str(e)}")
                return False
        
        # Function to actually run the segmentation
        def run_segmentation():
            global current_z
            """Run Cellpose segmentation on the current image"""
            # Get parameters
            cell_diameter = float(global_parameters['cell_diameter'])
            flow_threshold = float(global_parameters['flow_threshold'])
            cellprob_threshold = float(global_parameters['cellprob_threshold'])
            
            # Get the current processed image
            print("Segmenting on z-slice:", current_z, "channel:", segmentation_channel)
            if processed_slice is None:
                image_to_segment = update_slice(z_slice=current_z, channel=segmentation_channel)
            else:
                image_to_segment = apply_preprocessing(current_z, segmentation_channel)

            # Apply segmentation
            masks, outlines = apply_segmentation(image_to_segment, cell_diameter, flow_threshold, cellprob_threshold)
            
            # Add or update segmentation layers
            if "Segmentation" in viewer.layers:
                viewer.layers["Segmentation"].data = masks
            else:
                # Add as labels layer with random colors
                viewer.add_labels(masks, name="Segmentation", visible=False)
            
            # Add or update cell outlines layer
            if "Cell Outlines" in viewer.layers:
                viewer.layers["Cell Outlines"].data = outlines
            else:
                # Add binary outlines as labels layer
                viewer.add_labels(outlines.astype(np.uint8), name="Cell Outlines")
            
            print("Segmentation complete")

    # ====== STEP 9: SAVE PARAMETERS FUNCTION ======
    @magicgui(call_button="Save Parameters")
    def save_parameters():
        # Create a copy of the original dataframe to preserve structure
        updated_df = parameters_df.copy()
        
        # Update the 'Default Value' column with current values
        for index, row in updated_df.iterrows():
            parameter_name = row['parameter']
            channel = row['channel']
            
            # Check if this is a channel-specific parameter
            if pd.notnull(channel):
                try:
                    channel_num = int(channel)
                    if channel_num in channel_parameters and parameter_name in channel_parameters[channel_num]:
                        updated_df.at[index, 'value'] = channel_parameters[channel_num][parameter_name]
                except ValueError:
                    # Not a valid channel number, check global parameters
                    if parameter_name in global_parameters:
                        updated_df.at[index, 'value'] = global_parameters[parameter_name]
            else:
                # Global parameter
                if parameter_name in global_parameters:
                    updated_df.at[index, 'value'] = global_parameters[parameter_name]
        
        # Save to CSV
        try:
            updated_df.to_csv(output_parameters_path, index=False)
            print(f"Parameters successfully saved to {output_parameters_path}")
        except Exception as e:
            print(f"Error saving parameters to CSV: {str(e)}")
            raise

    # ====== STEP 10: ADD WIDGETS TO VIEWER ======
    # Add the slice selection
    viewer.window.add_dock_widget(update_slice, name="Z-slice and Channel")

    # Add the process selection widget
    viewer.window.add_dock_widget(process_container, name="Preprocessing")

    # Add the parameters widget
    viewer.window.add_dock_widget(processing_parameters, name="Preprocessing Parameters")

     # Add segmentation widget
    if segmentation_selected == True:
        viewer.window.add_dock_widget(segmentation_widget, name="Segmentation")

    # Add save button - keeping this button as saving should be deliberate
    viewer.window.add_dock_widget(save_parameters, name="Save Parameters")

    # Initialize the view with the first slice
    update_slice(z_slice=0, channel=0)

    # Start the application
    napari.run()

def run_analysis_window(progress_dialog):
    """Complete run analysis window with integrated pipeline"""
    from cellpose import models
    import napari
    import threading
    
    if not project_path:
        messagebox.showerror("Error", "Please load a project first.")
        return

    print("Initialising window...")
    preprocessed_image = None
    root_window = None

    # Create main window for this analysis
    root_window = tk.Tk()
    root_window.withdraw()  # Hide the main window
    
    # Create analysis options window
    analysis_window = tk.Toplevel(root_window)
    analysis_window.title("Run Analysis")
    analysis_window.geometry("400x300")

    frame = tk.Frame(analysis_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(frame, text="Run Analysis", font=("Arial", 12, "bold")).pack(pady=10)

    processes_frame = tk.LabelFrame(frame, text="Select Processes")
    processes_frame.pack(fill=tk.X, pady=10)

    processes = {
        "Preprocessing": tk.BooleanVar(value=True),
        "Segmentation": tk.BooleanVar(value=True),
        "Analysis": tk.BooleanVar(value=True)
    }

    for process, var in processes.items():
        tk.Checkbutton(processes_frame, text=process, variable=var).pack(anchor="w", padx=10)

    parameters_frame = tk.LabelFrame(frame, text="Parameters")
    parameters_frame.pack(fill=tk.X, pady=10)

    button_frame = tk.Frame(frame)
    button_frame.pack(pady=20)

    tk.Button(
        button_frame, 
        text="Cancel", 
        command=lambda: (analysis_window.destroy(), root_window.destroy()), 
        width=10
    ).pack(side=tk.LEFT, padx=5)
    
    def start_analysis(processes, root_window, progress_dialog):
        print("starting analysis")
        try:
            result = load_project_data(project_path, processes)
            if result is None:
                return
                
            selected_processes, channel_parameters, global_parameters, segmentation_channel_index, tiff_files, total_files = result

            progress_dialog = progressdialog(
                parent=root_window,
                title="Processing Images", 
                total_images=total_files,
            )

            processing_thread = threading.Thread(
                target=process_all_images,
                args=(project_path, 
                        selected_processes, 
                        channel_parameters,
                        global_parameters,        
                        segmentation_channel_index, 
                        tiff_files, 
                        total_files,
                        root_window,
                        progress_dialog
                        )
            )
            processing_thread.daemon = True
            processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")

    tk.Button(
        button_frame, 
        text="Start", 
        command=lambda: (analysis_window.destroy(), start_analysis(processes, root_window, progress_dialog)), 
        width=10
    ).pack(side=tk.LEFT, padx=5)
    
    root_window.mainloop()

def choose_processes_window():
    # Use os.path.join for cross-platform path handling
    processes_csv_path = os.path.join(project_path, 'processes.csv')
    if not loaded_project:
        messagebox.showerror("Error", "Please load a project first.")
        return
   
    process_window = tk.Toplevel(root)
    process_window.title("Choose Segmentation")
    process_window.geometry("400x300")  # Reduced height since we removed analysis options
   
    frame = tk.Frame(process_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
   
    tk.Label(frame, text="Select Segmentation Option", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Segmentation frame with radio buttons (mutually exclusive choices)
    segmentation_frame = tk.LabelFrame(frame, text="Segmentation")
    segmentation_frame.pack(fill=tk.X, pady=10)
   
    # Use a StringVar for radio buttons with "none" as default
    segmentation_choice = tk.StringVar(value="none")
   
    # Radio button options
    tk.Radiobutton(segmentation_frame, text="No segmentation",
                  variable=segmentation_choice, value="none").pack(anchor='w', pady=2)
    tk.Radiobutton(segmentation_frame, text="Cellpose nuclear segmentation",
                  variable=segmentation_choice, value="nuclear").pack(anchor='w', pady=2)
    tk.Radiobutton(segmentation_frame, text="Cellpose cellular segmentation",
                  variable=segmentation_choice, value="cellular").pack(anchor='w', pady=2)
   
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=20)
   
    def save_processes():
        # Get segmentation choice
        seg_choice = segmentation_choice.get()
        nuclear_selected = (seg_choice == "nuclear")
        cellular_selected = (seg_choice == "cellular")
       
        # Write to CSV
        with open(processes_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["category", "process", "selected"])
           
            # Write segmentation choices
            writer.writerow(["segmentation", "cellpose_nuclear_segmentation", "yes" if nuclear_selected else "no"])
            writer.writerow(["segmentation", "cellpose_cellular_segmentation", "yes" if cellular_selected else "no"])
       
        # Check if a segmentation process is selected (excluding "none")
        if seg_choice == "none":
            messagebox.showwarning("Warning", "No segmentation process selected.")
            return
       
        messagebox.showinfo("Process Saved", "Selected segmentation process has been saved to CSV.")
        process_window.destroy()
   
    tk.Button(button_frame, text="Cancel", command=process_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save", command=save_processes, width=10).pack(side=tk.LEFT, padx=5)

def load_project():
    global loaded_project, project_path
    
    project_path = filedialog.askdirectory(title="Select Project Folder")
    project_path = os.path.normpath(project_path)  # Normalize path
    if not project_path:
        return
        
    # Simple validation - just check if the folder exists
    if not os.path.isdir(project_path):
        messagebox.showerror("Error", "Invalid folder.")
        return
        
    loaded_project = True
    update_main_page()

def update_main_page():
    # Update project status label
    if loaded_project:
        project_name = os.path.basename(project_path)
        status_var.set(f"Project: {project_name}\nLocation: {project_path}")
        
        # Enable buttons
        for btn in [choose_processes_btn, define_parameters_btn, run_analysis_btn]:
            btn.config(state=tk.NORMAL)
    else:
        status_var.set("No project loaded")
        
        # Disable buttons
        for btn in [choose_processes_btn, define_parameters_btn, run_analysis_btn]:
            btn.config(state=tk.DISABLED)

def main_menu():
    global root, status_var
    global choose_processes_btn, define_parameters_btn, run_analysis_btn
    
    root = tk.Tk()
    root.title("Image Analysis")
    root.geometry("400x350")

    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header
    tk.Label(main_frame, text="Image Analysis Tool", font=("Arial", 14, "bold")).pack(pady=(0, 15))

    # Project buttons
    project_frame = tk.Frame(main_frame)
    project_frame.pack(fill=tk.X, pady=5)
    
    tk.Button(project_frame, text="Create Project", command=create_project_window, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(project_frame, text="Load Project", command=load_project, width=15).pack(side=tk.LEFT, padx=5)
    
    # Status display
    status_frame = tk.LabelFrame(main_frame, text="Project Status")
    status_frame.pack(fill=tk.X, pady=10, padx=5)
    
    status_var = tk.StringVar(value="No project loaded")
    status_label = tk.Label(status_frame, textvariable=status_var, justify=tk.LEFT, padx=10, pady=5)
    status_label.pack(fill=tk.X)
    
    # Action buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    choose_processes_btn = tk.Button(button_frame, text="Select segmentation options", command=choose_processes_window, width=42, state=tk.DISABLED)
    choose_processes_btn.pack(pady=3)
    
    define_parameters_btn = tk.Button(button_frame, text="Define Parameters", command=define_parameters_window, width=42, state=tk.DISABLED)
    define_parameters_btn.pack(pady=3)
    
    run_analysis_btn = tk.Button(button_frame, text="Run Analysis", command=run_analysis_window(progress_dialog), width=42, state=tk.DISABLED)
    run_analysis_btn.pack(pady=3)

    root.mainloop()

# Run the application
if __name__ == "__main__":
    main_menu()