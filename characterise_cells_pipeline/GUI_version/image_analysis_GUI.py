#%%

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
import tifffile

# Global variables
loaded_project = True

#%%
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
        path = filedialog.askopenfilename() or filedialog.askdirectory()
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
    import numpy as np
    import pandas as pd
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
        
        segmentation_rows = df[df['segmentation_channel'].str() == 'yes']
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
        segmentation_row = processes_df[processes_df['process'] == 'cellpose_nuclear_segmentation']
        segmentation_selected = len(segmentation_row) > 0 and segmentation_row['selected'].values[0] == 'yes'
    else:
        segmentation_selected = True  # Default to True if file doesn't exist
    
    # Load parameters from parameters_updated.csv
    parameters_path = os.path.join(project_root, 'parameters_updated.csv')
    if os.path.exists(parameters_path):
        parameters_df = pd.read_csv(parameters_path)
        
        # Extract parameters with correct data type
        params = {}
        for _, row in parameters_df.iterrows():
            param_name = row['parameter']
            param_value = row['value']
            data_type = row['data_type']
            
            if data_type == 'Float':
                params[param_name] = float(param_value)
            elif data_type == 'Integer':
                params[param_name] = int(param_value)
            else:
                params[param_name] = param_value
        
        # Extract specific parameters with defaults
        clahe_kernel_size = params.get('CLAHE_kernel_size', 64)
        clahe_clip_limit = params.get('CLAHE_clip_limit', 0.01)
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
    
    # Determine segmentation channel from channel_details.csv
    channel_path = os.path.join(project_root, 'channel_details.csv')
    segmentation_channel_idx = 0  # Default to first channel
    
    if os.path.exists(channel_path):
        channel_df = pd.read_csv(channel_path)
        segmentation_rows = channel_df[channel_df['segmentation_channel'] == 'yes']
        
        if len(segmentation_rows) > 0:
            segmentation_channel_idx = segmentation_rows.index[0]
    
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
              f"Cellprob threshold = {cellprob_threshold}, Segmentation Channel = {segmentation_channel_idx}")
        
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
                
                # Extract segmentation channel
                if len(image_stack.shape) > 3:  # If multi-channel
                    segmentation_slice = image_stack[:, segmentation_channel_idx, :, :]
                else:  # Single channel
                    segmentation_slice = image_stack
                
                preprocessed_image = None
                segmented_image = None
                
                # Run preprocessing if selected
                if "Preprocessing" in selected_processes:
                    print(f"{datetime.now():%H:%M:%S} - Starting preprocessing")
                    preprocessed_image = apply_preprocessing(segmentation_slice)
                    
                    # Save preprocessed image
                    preprocessed_path = os.path.join(output_folder, f"{os.path.splitext(tiff_file)[0]}_preprocessed.tiff")
                    tifffile.imwrite(preprocessed_path, preprocessed_image)
                    print(f"Saved preprocessed image to {preprocessed_path}")
                
                # Run segmentation if selected
                if "Segmentation" in selected_processes and segmentation_selected:
                    print(f"{datetime.now():%H:%M:%S} - Starting segmentation")
                    
                    # Use preprocessed image if available, otherwise use raw segmentation channel
                    seg_input = preprocessed_image if preprocessed_image is not None else segmentation_slice
                    
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
    
    tk.Label(params_frame, text=f"Segmentation Channel: {segmentation_channel_idx}").pack(anchor="w", padx=10)
    if segmentation_selected:
        tk.Label(params_frame, text=f"Cell Diameter: {cell_diameter}").pack(anchor="w", padx=10)
    
    # Buttons
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="Cancel", command=analysis_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Start", command=run_full_analysis, width=10).pack(side=tk.LEFT, padx=5)

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
    
    #messagebox.showinfo("Project Loaded", f"Project loaded from {project_path}")

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
    
    choose_processes_btn = tk.Button(button_frame, text="Select segmentation options", command=choose_processes_window, width=42, state=tk.DISABLED)
    choose_processes_btn.pack(pady=3)
    
    define_parameters_btn = tk.Button(button_frame, text="Define Parameters", command=define_parameters_window, width=42, state=tk.DISABLED)
    define_parameters_btn.pack(pady=3)
    
    run_analysis_btn = tk.Button(button_frame, text="Run Analysis", command=run_analysis_window, width=42, state=tk.DISABLED)
    run_analysis_btn.pack(pady=3)

    root.mainloop()

#%%
main_menu()

#%%
#input_image_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris_cropped.tiff"
#input_image_path = "/Users/oskar/Desktop/format_test/not needed/SBSO_stellaris.tiff"

#image = tifffile.imread(input_image_path)
#define_parameters_window(image)
# %%
