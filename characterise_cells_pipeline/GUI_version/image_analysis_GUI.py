import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv

# Global variables
loaded_project = True
project_path = "/Users/oskar/Desktop/format_test"
foldername = "/Users/oskar/Desktop/format_test"

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

        if contains_nuclear_var.get():
            nuclear_index = nuclear_var.get()
            if nuclear_index == -1 or not channel_names[nuclear_index]:
                messagebox.showerror("Input Error", "Nuclear channel must be a valid, non-empty channel.")
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
                writer.writerow(["Channel", "Nuclear Channel"])
                for idx, name in enumerate(channel_names):
                    if name:  # Only write non-empty channel names
                        writer.writerow([name, 'Yes' if nuclear_var.get() == idx else 'No'])
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
        path = filedialog.askopenfilename() or filedialog.askdirectory()
        if path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, path)

    tk.Button(input_frame, text="Browse", command=browse_input).pack(side=tk.LEFT)
    
    # Destination folder
    dest_frame = tk.Frame(frame)
    dest_frame.pack(fill=tk.X, pady=5)
    tk.Label(dest_frame, text="Destination Folder:").pack(side=tk.LEFT)
    dest_folder_entry = tk.Entry(dest_frame, width=40)
    dest_folder_entry.pack(side=tk.LEFT, padx=5)
    
    def browse_destination():
        foldername = filedialog.askdirectory()
        if foldername:
            dest_folder_entry.delete(0, tk.END)
            dest_folder_entry.insert(0, foldername)

    tk.Button(dest_frame, text="Browse", command=browse_destination).pack(side=tk.LEFT)
    
    # Nuclear channel option
    nuclear_frame = tk.Frame(frame)
    nuclear_frame.pack(fill=tk.X, pady=5)
    
    contains_nuclear_var = tk.BooleanVar()
    tk.Checkbutton(nuclear_frame, text="Contains Nuclear Channel", variable=contains_nuclear_var).pack(anchor="w")
    
    nuclear_var = tk.IntVar(value=-1)
    
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
        
        radio = tk.Radiobutton(channel_frame, text="Nuclear", variable=nuclear_var, value=i)
        radio.pack(side=tk.LEFT)
        radio_buttons.append(radio)
        
        # Initially disable all radio buttons
        radio.config(state=tk.DISABLED)
    
    # Function to toggle radio buttons
    def toggle_nuclear_options():
        state = tk.NORMAL if contains_nuclear_var.get() else tk.DISABLED
        for rb in radio_buttons:
            rb.config(state=state)
    
    # Bind checkbox to toggle function
    contains_nuclear_var.trace("w", lambda *args: toggle_nuclear_options())
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Cancel", command=project_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Create Project", command=create_project, width=15).pack(side=tk.LEFT, padx=5)

def define_parameters_window():

    loaded_project = True
    project_path = foldername  
            
    if not loaded_project:
        messagebox.showerror("Error", "Please load a project first.")
        return
    
    input_image_path = filedialog.askopenfilename(title="Select Sample Image File")

    print(input_image_path)

    import numpy as np
    import tifffile
    import pandas as pd
    import os
    from skimage.exposure import equalize_adapthist
    from skimage import img_as_ubyte
    import napari
    from magicgui import magicgui
    from cv2 import GaussianBlur
    from magicgui.widgets import Container, CheckBox
    from cellpose import models
    from skimage.segmentation import find_boundaries

    # Store current slice information as global variables for easy access
   #current_z = 14
    #current_channel = 0
    #processed_image = None
    segmentation_selected = True

    # ====== STEP 1: FILE PATHS AND ERROR CHECKING ======
    #input_image_path = '/Users/oskar/Desktop/format_test/test.lsm'
    input_parameters_csv = '/Users/oskar/Desktop/format_test/parameters.csv'

    # ====== STEP 2: LOAD DATA ======
    print("INPUT PATH - ", input_image_path)
    try:
        # Load image
        print(input_image_path)
        image = tifffile.imread(input_image_path)
        print("Successfully loaded")
        print("Image loaded with shape (z, c, y, x):", image.shape, "dtype:", image.dtype)
        
        if len(image.shape) != 4:
            raise ValueError(f"Expected 4D image (z, c, y, x), got shape {image.shape}")
        
        total_z = image.shape[0]
        total_channels = image.shape[1]
        min_xy_size = min(image.shape[2], image.shape[3])
        
        # Load parameters file
        try:
            parameters_df = pd.read_csv(input_parameters_csv)
            required_columns = ['Parameter', 'Process', 'Default Value', 'Min', 'Max']
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

    # Dictionary to store current parameter values
    param_values = {}

    # Dictionary to store process selection state
    process_enabled = {process: False for process in available_processes}

    # Initialize parameter values from the CSV
    for _, row in parameters_df.iterrows():
        param_name = row['Parameter']
        default_value = row['Default Value']
        # Use default if value is NaN
        param_values[param_name] = default_value

    # Extract min and max values for each parameter from CSV
    param_min_max = {}
    for _, row in parameters_df.iterrows():
        param_name = row['Parameter']
        param_min_max[param_name] = {
            'min': row['Min'],
            'max': row['Max']
        }

    # ====== STEP 6: PREPROCESSING FUNCTION ======
    def apply_preprocessing(current_z, current_channel):
        """Apply selected preprocessing steps to the current slice"""
        global processed_image
        
        try:
            # Get the original slice using global variables
            original_slice = image[current_z, current_channel, :, :]
            processed = original_slice.copy()
            
            # Get selected processes from the global process_enabled dictionary
            selected_processes = [process for process, enabled in process_enabled.items() if enabled]
                        
            # Apply processing steps
            for process in selected_processes:
                if process == 'CLAHE':
                    kernel_size = int(param_values.get('CLAHE_kernel_size', 16))
                    clip_limit = float(param_values.get('CLAHE_cliplimit', 0.005))
                    n_bins = int(param_values.get('CLAHE_n_bins', 11))
                    #print(f"Applying CLAHE with kernel_size={kernel_size}, clip_limit={clip_limit}, n_bins={n_bins}")
                    processed = apply_CLAHE(processed, kernel_size, clip_limit, n_bins)
                    
                elif process == 'Gaussian':
                    kernel_size = int(param_values.get('gaussian_kernel_size', 11))
                    sigma = float(param_values.get('gaussian_sigma', 0.4))
                    processed = apply_Gaussian(processed, kernel_size, sigma)
        
            # Store the processed image for segmentation
            processed_image = processed
            
            # Update viewer
            if "Preprocessed" in viewer.layers:
                viewer.layers["Preprocessed"].data = processed
            else:
                viewer.add_image(processed, name="Preprocessed")

            return processed_image
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            return False

    # ====== STEP 5: SLICE SELECTION WIDGET ======
    @magicgui(
        z_slice={"label": "Z slice", "widget_type": "Slider", "min": 0, "max": total_z-1},
        channel={"label": "Channel", "widget_type": "Slider", "min": 0, "max": total_channels-1}, auto_call=True
    )
    def update_slice(z_slice=14, channel=0):
        """Select a z-slice and channel to view"""
        global current_z, current_channel
        
        try:
            # Update global variables
            current_z = z_slice
            current_channel = channel
            
            # Get the original slice
            original_slice = image[z_slice, channel, :, :]
            
            # Update viewer
            if "unprocessed" in viewer.layers:
                viewer.layers["unprocessed"].data = original_slice
            else:
                viewer.add_image(original_slice, name="unprocessed")
                
            # Automatically apply preprocessing after changing slice
            preprocessed_image = apply_preprocessing(current_z, current_channel)

            return preprocessed_image
        except Exception as e:
            print(f"Error selecting slice: {str(e)}")
            return False

    # ====== STEP 7: CREATE PARAMETER WIDGETS ======
    # Process selection widget
    process_container = Container(layout="vertical")

    # Function to handle process checkbox changes
    def on_process_toggle(process_name, value):
        process_enabled[process_name] = value
        # Apply preprocessing to update the image
        update_slice()

    # Add checkboxes for each process
    for process in available_processes:
        checkbox = CheckBox(name=f"Enable {process}", value=process_enabled[process])
        # Use lambda with default args to avoid late binding issues
        checkbox.changed.connect(lambda v, p=process: on_process_toggle(p, v))
        process_container.append(checkbox)

    # Get min/max values for each parameter from the parameters_df
    clahe_kernel_min = int(param_min_max.get('CLAHE_kernel_size', {}).get('min', 3))
    clahe_kernel_max = int(param_min_max.get('CLAHE_kernel_size', {}).get('max', 100))
    
    clahe_clip_min = float(param_min_max.get('CLAHE_cliplimit', {}).get('min', 0.0001))
    clahe_clip_max = float(param_min_max.get('CLAHE_cliplimit', {}).get('max', 0.5))
    
    clahe_bins_min = int(param_min_max.get('CLAHE_n_bins', {}).get('min', 3))
    clahe_bins_max = int(param_min_max.get('CLAHE_n_bins', {}).get('max', 1000))
    
    gaussian_kernel_min = int(param_min_max.get('gaussian_kernel_size', {}).get('min', 3))
    gaussian_kernel_max = int(param_min_max.get('gaussian_kernel_size', {}).get('max', 31))
    
    gaussian_sigma_min = float(param_min_max.get('gaussian_sigma', {}).get('min', 0.05))
    gaussian_sigma_max = float(param_min_max.get('gaussian_sigma', {}).get('max', 5))

    # Combined parameters widget for all processing techniques
    @magicgui(
        # CLAHE parameters
        clahe_kernel_size={"label": "CLAHE Kernel Size", "widget_type": "Slider", 
                        "min": clahe_kernel_min, "max": min_xy_size, "step": 1},
        clahe_clip_limit={"label": "CLAHE Clip Limit", "widget_type": "FloatSlider", 
                        "min": 0.1, "max": 1.0, "step": 0.01},
        clahe_n_bins={"label": "CLAHE n bins", "widget_type": "Slider", 
                        "min": clahe_bins_min, "max": clahe_bins_max, "step": 1},

        # Gaussian parameters
        gaussian_kernel_size={"label": "Gaussian Kernel Size", "widget_type": "Slider", 
                            "min": gaussian_kernel_min, "max": min_xy_size, "step": 2},
        gaussian_sigma={"label": "Gaussian Sigma", "widget_type": "FloatSlider", 
                    "min": 0.1, "max": 1.0, "step": 0.001}, auto_call=True
    )
    def processing_parameters(
        clahe_kernel_size=16, 
        clahe_clip_limit=0.5,
        clahe_n_bins=11,
        gaussian_kernel_size=11, 
        gaussian_sigma=0.4
    ):
        try:
            # Map slider values logarithmically for specific parameters
            # For CLAHE clip limit (logarithmic mapping)
            clahe_clip_limit_mapped = clahe_clip_min + (clahe_clip_max - clahe_clip_min) * (np.log10(10 * clahe_clip_limit))
            
            # For Gaussian sigma (logarithmic mapping)
            gaussian_sigma_mapped = gaussian_sigma_min + (gaussian_sigma_max - gaussian_sigma_min) * (np.log10(10 * gaussian_sigma))

            print("gaussian sigma")
            print("gaussian sigma mapped", gaussian_sigma_mapped)
            print("clahe clip limit")
            print("clahe clip limit mapped", clahe_clip_limit_mapped)

            # Update parameters with mapped values
            param_values['CLAHE_kernel_size'] = clahe_kernel_size
            param_values['CLAHE_cliplimit'] = clahe_clip_limit_mapped
            param_values['CLAHE_n_bins'] = clahe_n_bins
            
            # Update Gaussian parameters (ensure kernel is odd)
            if gaussian_kernel_size % 2 == 0:
                gaussian_kernel_size += 1
            param_values['gaussian_kernel_size'] = gaussian_kernel_size
            param_values['gaussian_sigma'] = gaussian_sigma_mapped
            
            # Re-apply preprocessing with updated parameters
            update_slice()

            return True
        except Exception as e:
            print(f"Error updating parameters: {str(e)}")
            return False

    # ====== STEP 8: SEGMENTATION WIDGET AND FUNCTION ======
    # Segmentation widget without run button
    if segmentation_selected == True:
        # Get min/max values for segmentation parameters
        cell_diam_min = float(param_min_max.get('cell_diameter', {}).get('min', 5.0))
        cell_diam_max = float(param_min_max.get('cell_diameter', {}).get('max', 50.0))
        
        flow_thresh_min = float(param_min_max.get('flow_threshold', {}).get('min', 0.1))
        flow_thresh_max = float(param_min_max.get('flow_threshold', {}).get('max', 1.0))
        
        cell_prob_min = float(param_min_max.get('cellprob_threshold', {}).get('min', 0.0))
        cell_prob_max = float(param_min_max.get('cellprob_threshold', {}).get('max', 10.0))
        
        @magicgui(
            cell_diameter={"label": "Cell diameter", "widget_type": "FloatSlider", 
                        "min": cell_diam_min, "max": cell_diam_max, "step": 0.5},
            flow_threshold={"label": "Flow threshold", "widget_type": "FloatSlider", 
                        "min": flow_thresh_min, "max": flow_thresh_max, "step": 0.01},
            cellprob_threshold={"label": "Cell probability threshold", "widget_type": "FloatSlider", 
                            "min": cell_prob_min, "max": cell_prob_max, "step": 0.1}
        )
        def segmentation_widget(cell_diameter=8.0, flow_threshold=0.5, cellprob_threshold=0.5):
            """Set parameters for Cellpose segmentation"""
            try:
                # Store parameters
                param_values['cell_diameter'] = cell_diameter
                param_values['flow_threshold'] = flow_threshold
                param_values['cellprob_threshold'] = cellprob_threshold
                
                # Run segmentation with new parameters
                run_segmentation()
                
                return True
            except Exception as e:
                print(f"Error setting segmentation parameters: {str(e)}")
                return False
        
        # Function to actually run the segmentation
        def run_segmentation():
            """Run Cellpose segmentation on the current image"""
            try:
                # Get parameters
                cell_diameter = float(param_values.get('cell_diameter', 8.0))
                flow_threshold = float(param_values.get('flow_threshold', 0.5))
                cellprob_threshold = float(param_values.get('cellprob_threshold', 0.5))
                
                # Get the current processed image
                if processed_image is None:
                    image_to_segment = update_slice()
                else:
                    image_to_segment = processed_image

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
                
                print("Segmentation complete and displayed")
                return True
            except Exception as e:
                print(f"Error during segmentation: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

    # ====== STEP 9: SAVE PARAMETERS FUNCTION ======
    @magicgui(call_button="Save Parameters")
    def save_parameters():
        try:
            # Create a copy of the parameters_df to modify
            updated_df = parameters_df.copy()
            
            # Ensure 'Enabled' column exists in the dataframe
            if 'Enabled' not in updated_df.columns:
                updated_df['Enabled'] = 'FALSE'
            
            # Update parameters DataFrame with current values
            for param_name, value in param_values.items():
                mask = updated_df['Parameter'] == param_name
                if any(mask):
                    # Update 'Value' column if it exists, otherwise add it
                    if 'Value' not in updated_df.columns:
                        updated_df['Value'] = np.nan
                    updated_df.loc[mask, 'Value'] = value
                    
                    # Update the 'Enabled' column based on process_enabled dictionary
                    process = updated_df.loc[mask, 'Process'].iloc[0]
                    if process in process_enabled:
                        updated_df.loc[mask, 'Enabled'] = str(process_enabled[process]).upper()
            
            # Save to CSV
            output_params_path = os.path.join(os.path.dirname(input_parameters_csv), 'parameters_updated.csv')
            updated_df.to_csv(output_params_path, index=False)
            
            print(f"Parameters saved successfully to {output_params_path}")
            
            # Close Napari viewer
            viewer.close()
            
            return True
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            return False

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
    update_slice()

    # Start the application
    napari.run()

def run_analysis_window():
    import os
    import numpy as np
    import tkinter as tk
    from tkinter import messagebox
    from datetime import datetime
    from skimage import img_as_ubyte
    from skimage.exposure import equalize_adapthist
    from skimage.filters import gaussian
    import tifffile
    from cellpose import models
    import pandas as pd
    import napari
    
    if not loaded_project:
        messagebox.showerror("Error", "Please load a project first.")
        return
        
    # Load project parameters and check if segmentation is selected
    project_root = os.path.dirname(os.path.abspath(loaded_project))
    
    # Check if segmentation is selected from processes.csv
    processes_path = os.path.join(project_root, 'processes.csv')
    if os.path.exists(processes_path):
        processes_df = pd.read_csv(processes_path)
        segmentation_row = processes_df[processes_df['Process'] == 'Cellpose Nuclear Segmentation']
        segmentation_selected = len(segmentation_row) > 0 and segmentation_row['Selected'].values[0] == 'Yes'
    else:
        segmentation_selected = True  # Default to True if file doesn't exist
    
    # Load parameters from parameters_updated.csv
    parameters_path = os.path.join(project_root, 'parameters_updated.csv')
    if os.path.exists(parameters_path):
        parameters_df = pd.read_csv(parameters_path)
        
        # Extract parameters with correct data type
        params = {}
        for _, row in parameters_df.iterrows():
            param_name = row['Parameter']
            param_value = row['Value']
            data_type = row['Data type']
            
            if data_type == 'Float':
                params[param_name] = float(param_value)
            elif data_type == 'Integer':
                params[param_name] = int(param_value)
            else:
                params[param_name] = param_value
        
        # Extract specific parameters with defaults
        clahe_kernel_size = params.get('CLAHE_kernel_size', 64)
        clahe_clip_limit = params.get('CLAHE_cliplimit', 0.01)
        clahe_n_bins = params.get('CLAHE_n_bins', 256)
        gaussian_kernel_size = params.get('gaussian_kernel_size', 3)
        gaussian_sigma = params.get('gaussian_sigma', 1.0)
        cell_diameter = params.get('cell_diameter', 30)
        flow_threshold = params.get('flow_threshold', 0.5)
        cellprob_threshold = params.get('cellprob_threshold', 0.1)
        iou_threshold = params.get('iou_threshold', 0.5)
    else:
        # Default parameters if file doesn't exist
        clahe_kernel_size = 64
        clahe_clip_limit = 0.01
        clahe_n_bins = 256
        gaussian_kernel_size = 3
        gaussian_sigma = 1.0
        cell_diameter = 30
        flow_threshold = 0.5
        cellprob_threshold = 0.1
        iou_threshold = 0.5
    
    # Determine nuclear channel from channel_details.csv
    channel_path = os.path.join(project_root, 'channel_details.csv')
    nuclear_channel_idx = 0  # Default to first channel
    
    if os.path.exists(channel_path):
        channel_df = pd.read_csv(channel_path)
        nuclear_rows = channel_df[channel_df['Nuclear Channel'] == 'Yes']
        
        if len(nuclear_rows) > 0:
            # Get the index from the row index (B2 -> index 0, B6 -> index 4)
            # Assuming B2, B3, B4, etc. correspond to rows 1, 2, 3, etc.
            nuclear_channel_idx = nuclear_rows.index[0]
    
    def apply_preprocessing(image):
        """Apply Gaussian blur and CLAHE preprocessing to image"""
        # Apply Gaussian blur
        blurred = gaussian(image, sigma=gaussian_sigma, truncate=gaussian_kernel_size/2)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        preprocessed = img_as_ubyte(equalize_adapthist(
            blurred, 
            kernel_size=clahe_kernel_size, 
            clip_limit=clahe_clip_limit, 
            nbins=clahe_n_bins
        ))
        
        return preprocessed
    
    def segment_2D(channel_slice):
        """Segment nuclei in 2D slices using Cellpose"""
        model = models.Cellpose(gpu=False, model_type='nuclei')
        
        total_z = channel_slice.shape[0]
        segmented_image = np.zeros_like(channel_slice, dtype=np.uint16)
        total_cells_segmented = 0
        
        for z in range(total_z):
            if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
                print(f"{datetime.now():%H:%M:%S} - Segmentation {((z + 1) * 100) // total_z}% complete")
    
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
        print(f"Cell diameter = {cell_diameter}, Flow threshold = {flow_threshold}, "
              f"Cellprob threshold = {cellprob_threshold}, Nuclear Channel = {nuclear_channel_idx}")
        
        return segmented_image
    
    def calculate_iou(cell_1, cell_2):
        """Calculate Intersection over Union (IoU) between two binary masks"""
        intersection = np.logical_and(cell_1, cell_2).sum()
        union = np.logical_or(cell_1, cell_2).sum()
        if union == 0:
            return 0
        return intersection / union
    
    def stitch_by_iou(segmented_image):
        """Stitch segmented cells across z-slices based on IoU"""
        total_z = segmented_image.shape[0]
        
        def relabel_stitched_masks(segmented_image):
            """Relabels the stitched 2D segmentation masks based on IoU across z-slices"""
            stitched_image = np.squeeze(segmented_image).astype(np.uint16)
             
            current_label = 1
            for z in range(1, total_z):
                if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
                    print(f"{datetime.now():%H:%M:%S} - Stitching {((z + 1) * 100) // total_z}% complete)")
    
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
    
    def run_full_analysis():
        """Main function to run the selected analysis processes"""
        # Check if project folder has TIFF files
        tiff_files = [f for f in os.listdir(project_root) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
        
        if not tiff_files:
            messagebox.showerror("Error", "No TIFF files found in the project folder.")
            return
        
        # Get selected processes
        selected_processes = [p for p, v in processes.items() if v.get()]
        if not selected_processes:
            messagebox.showerror("Error", "Please select at least one process.")
            return
        
        print(f"Starting analysis with processes: {', '.join(selected_processes)}")
        
        for tiff_file in tiff_files:
            tiff_path = os.path.join(project_root, tiff_file)
            output_folder = os.path.join(project_root, os.path.splitext(tiff_file)[0])
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Load the TIFF file
            try:
                image_stack = tifffile.imread(tiff_path)
                print(f"Loaded image with shape: {image_stack.shape}")
                
                # Extract nuclear channel
                if len(image_stack.shape) > 3:  # If multi-channel
                    nuclear_slice = image_stack[:, nuclear_channel_idx, :, :]
                else:  # Single channel
                    nuclear_slice = image_stack
                
                preprocessed_image = None
                segmented_image = None
                
                # Run preprocessing if selected
                if "Preprocessing" in selected_processes:
                    print(f"{datetime.now():%H:%M:%S} - Starting preprocessing")
                    preprocessed_image = apply_preprocessing(nuclear_slice)
                    
                    # Save preprocessed image
                    preprocessed_path = os.path.join(output_folder, f"{os.path.splitext(tiff_file)[0]}_preprocessed.tiff")
                    tifffile.imwrite(preprocessed_path, preprocessed_image)
                    print(f"Saved preprocessed image to {preprocessed_path}")
                
                # Run segmentation if selected
                if "Segmentation" in selected_processes and segmentation_selected:
                    print(f"{datetime.now():%H:%M:%S} - Starting segmentation")
                    
                    # Use preprocessed image if available, otherwise use raw nuclear channel
                    seg_input = preprocessed_image if preprocessed_image is not None else nuclear_slice
                    
                    # Run 2D segmentation
                    segmented_image = segment_2D(seg_input)
                    
                    # Run 3D stitching and cleaning
                    stitched_image = stitch_by_iou(segmented_image)
                    final_segmentation = clean_labels(stitched_image)
                    
                    # Save segmentation results
                    segmentation_path = os.path.join(output_folder, f"{os.path.splitext(tiff_file)[0]}_segmentation.tiff")
                    tifffile.imwrite(segmentation_path, final_segmentation)
                    print(f"Saved segmentation to {segmentation_path}")
                    
                    # Store for analysis and visualization
                    segmented_image = final_segmentation
                
                # Run analysis if selected
                if "Analysis" in selected_processes:
                    print(f"{datetime.now():%H:%M:%S} - Starting analysis")
                    # Here you would implement the analysis logic
                    # This part would depend on what specific analysis you want to perform
                    
                # Open in napari for visual confirmation
                if segmented_image is not None or preprocessed_image is not None:
                    viewer = napari.Viewer()
                    
                    if preprocessed_image is not None:
                        viewer.add_image(preprocessed_image, name='Preprocessed')
                    
                    if segmented_image is not None:
                        viewer.add_labels(segmented_image, name='Segmentation')
                    
                    napari.run()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing {tiff_file}: {str(e)}")
                print(f"Error processing {tiff_file}: {str(e)}")
    
    # Create the GUI for analysis options
    analysis_window = tk.Toplevel(root)
    analysis_window.title("Run Analysis")
    analysis_window.geometry("400x300")
    
    frame = tk.Frame(analysis_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(frame, text="Run Analysis", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Process selection
    processes_frame = tk.LabelFrame(frame, text="Select Processes")
    processes_frame.pack(fill=tk.X, pady=10)
    
    processes = {
        "Preprocessing": tk.BooleanVar(value=True),
        "Segmentation": tk.BooleanVar(value=segmentation_selected),
        "Analysis": tk.BooleanVar(value=True)
    }
    
    for process, var in processes.items():
        tk.Checkbutton(processes_frame, text=process, variable=var).pack(anchor="w", padx=10)
    
    # Parameter display
    params_frame = tk.LabelFrame(frame, text="Parameters")
    params_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(params_frame, text=f"Nuclear Channel: {nuclear_channel_idx}").pack(anchor="w", padx=10)
    if segmentation_selected:
        tk.Label(params_frame, text=f"Cell Diameter: {cell_diameter}").pack(anchor="w", padx=10)
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="Cancel", command=analysis_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Start", command=run_full_analysis, width=10).pack(side=tk.LEFT, padx=5)

def choose_processes_window():
    processes_csv_path = foldername + '/processes.csv'
    if not loaded_project:
        messagebox.showerror("Error", "Please load a project first.")
        return
    
    process_window = tk.Toplevel(root)
    process_window.title("Choose Processes")
    process_window.geometry("350x450")
    
    frame = tk.Frame(process_window, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(frame, text="Select Processing Options", font=("Arial", 12, "bold")).pack(pady=10)
    options_frame = tk.Frame(frame)
    options_frame.pack(fill=tk.BOTH, expand=True)
    
    # Segmentation frame with radio buttons (mutually exclusive choices)
    segmentation_frame = tk.LabelFrame(options_frame, text="Segmentation")
    segmentation_frame.pack(fill=tk.X, pady=5)
    
    # Use a StringVar for radio buttons with an empty default
    segmentation_choice = tk.StringVar(value="none")
    
    # Add a "None" option to allow deselecting both
    tk.Radiobutton(segmentation_frame, text="No segmentation", 
                  variable=segmentation_choice, value="none").pack(anchor='w')
    tk.Radiobutton(segmentation_frame, text="Cellpose nuclear segmentation", 
                  variable=segmentation_choice, value="nuclear").pack(anchor='w')
    tk.Radiobutton(segmentation_frame, text="Cellpose cellular segmentation", 
                  variable=segmentation_choice, value="cellular").pack(anchor='w')
    
    # Analysis frame with checkboxes (multiple selections allowed)
    analysis_frame = tk.LabelFrame(options_frame, text="Analysis")
    analysis_frame.pack(fill=tk.X, pady=5)
    
    analysis_options = {
        "Intensity Analysis": tk.BooleanVar(),
        "Shape Analysis": tk.BooleanVar()
    }
    
    for option, var in analysis_options.items():
        tk.Checkbutton(analysis_frame, text=option, variable=var).pack(anchor="w")
    
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)
    
    def save_processes():
        selected = []
        
        # Get segmentation choice
        seg_choice = segmentation_choice.get()
        nuclear_selected = (seg_choice == "nuclear")
        cellular_selected = (seg_choice == "cellular")
        
        # Write to CSV
        with open(processes_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Process", "Selected"])
            
            # Write segmentation choices
            writer.writerow(["Segmentation", "Cellpose nuclear segmentation", "Yes" if nuclear_selected else "No"])
            writer.writerow(["Segmentation", "Cellpose cellular segmentation", "Yes" if cellular_selected else "No"])
            
            # Write analysis choices
            for option, var in analysis_options.items():
                selected.append(("Analysis", option, var.get()))
                writer.writerow(["Analysis", option, "Yes" if var.get() else "No"])
        
        # Check if any process is selected
        if not (nuclear_selected or cellular_selected or any(var.get() for var in analysis_options.values())):
            messagebox.showwarning("Warning", "No processes selected.")
            return
        
        messagebox.showinfo("Processes Saved", "Selected processes have been saved to CSV.")
        process_window.destroy()
    
    tk.Button(button_frame, text="Cancel", command=process_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save", command=save_processes, width=10).pack(side=tk.LEFT, padx=5)

def load_project():
    global loaded_project, project_path, foldername
    '''
    foldername = filedialog.askdirectory(title="Select Project Folder")
    if not foldername:
        return
        
    # Simple validation - just check if the folder exists
    if not os.path.isdir(foldername):
        messagebox.showerror("Error", "Invalid folder.")
        return
        
    loaded_project = True
    project_path = foldername
    '''
    loaded_project = True
    project_path = "/Users/oskar/Desktop/format_test"
    foldername = "/Users/oskar/Desktop/format_test"

    update_main_page()
    
    #messagebox.showinfo("Project Loaded", f"Project loaded from {foldername}")

def update_main_page():
    print(loaded_project)
    print(project_path)
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
    
    choose_processes_btn = tk.Button(button_frame, text="Choose Processes", command=choose_processes_window, width=42, state=tk.DISABLED)
    choose_processes_btn.pack(pady=3)
    
    define_parameters_btn = tk.Button(button_frame, text="Define Parameters", command=define_parameters_window, width=42, state=tk.DISABLED)
    define_parameters_btn.pack(pady=3)
    
    run_analysis_btn = tk.Button(button_frame, text="Run Analysis", command=run_analysis_window, width=42, state=tk.DISABLED)
    run_analysis_btn.pack(pady=3)

    root.mainloop()


main_menu()
