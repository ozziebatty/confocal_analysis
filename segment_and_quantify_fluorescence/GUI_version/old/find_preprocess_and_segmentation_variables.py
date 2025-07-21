print("Running")

import numpy as np
import tifffile
import pandas as pd
import os
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import napari
from magicgui import magicgui
from cellpose import models
from datetime import datetime
from skimage.segmentation import find_boundaries
from cv2 import GaussianBlur

# File paths
input_image_path = '/Users/oskar/Desktop/format_test/test.lsm'
input_processes_csv = '/Users/oskar/Desktop/format_test/processes.csv'
input_parameters_csv = '/Users/oskar/Desktop/format_test/parameters.csv'

# Load CSV configuration files
def load_processes():
    if not os.path.exists(input_processes_csv):
        raise ValueError("Processes file is missing")
    
    processes_df = pd.read_csv(input_processes_csv)

    return processes_df

def load_parameters():
    if not os.path.exists(input_parameters_csv):
        raise ValueError("Parameters file is missing")
    
    parameters_df = pd.read_csv(input_parameters_csv)
        
    return parameters_df

# Load image
image = tifffile.imread(input_image_path)
print("Original image shape (z, c, y, x):", image.shape, "dtype:", image.dtype)
total_z = image.shape[0]
total_channels = image.shape[1]

# Napari Viewer
viewer = napari.Viewer()

def apply_CLAHE(channel_z_slice, kernel_size, clip_limit, nbins=256):
    return img_as_ubyte(equalize_adapthist(channel_z_slice, 
                                         kernel_size=(kernel_size, kernel_size), 
                                         clip_limit=clip_limit, 
                                         nbins=nbins))

def apply_Gaussian(channel_z_slice, kernel_size, sigma):
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return GaussianBlur(channel_z_slice, ksize=(kernel_size, kernel_size), sigmaX=sigma)

# Load configurations
processes_df = load_processes()
parameters_df = load_parameters()
params_df = parameters_df

print(parameters_df)
print(processes_df)

# Create a dictionary of UI parameter values that will be updated by the widgets
ui_param_values = {}

# Store currently selected processes
selected_processes = processes_df[processes_df['Selected'] == 'Yes']['Process'].tolist()
print(selected_processes)

# Create the preprocessing UI
@magicgui(
    example_z={"label": "Z slice", "min": 0, "max": total_z - 1, "step": 1},
    channel={"label": "Channel", "min": 0, "max": total_channels - 1, "step": 1},
)
@magicgui(
    example_z={"label": "Z slice", "min": 0, "max": total_z - 1, "step": 1},
    channel={"label": "Channel", "min": 0, "max": total_channels - 1, "step": 1},
)
def update_preprocessing(example_z: int = 5, channel: int = 3):  # Remove **kwargs
    # Rest of the function remains the same
    print("apple")
    # Extract single-channel slice
    channel_z_slice = image[example_z, channel, :, :]
    processed = channel_z_slice.copy()
    print("banana")
    
    # Apply selected preprocessing steps in sequence
    for process in selected_processes:
        if process == 'CLAHE':
            print("generating CLAHE")
            kernel_size = ui_param_values.get('CLAHE_kernel_size', 16)
            clip_limit = ui_param_values.get('CLAHE_cliplimit', 0.005)
            processed = apply_CLAHE(processed, kernel_size, clip_limit)
        
        if process == 'Gaussian':
            print("generating Gaussian")  # Fixed typo "CGaussian"
            kernel_size = ui_param_values.get('Gaussian_kernel_size', 11)
            sigma = ui_param_values.get('Gaussian_sigma', 0.1)
            processed = apply_Gaussian(processed, kernel_size, sigma)

    if "Original" in viewer.layers:
        viewer.layers["Original"].data = channel_z_slice
    else:
        viewer.add_image(channel_z_slice, name="Original")
        
    if "Processed" in viewer.layers:
        viewer.layers["Processed"].data = processed
    else:
        viewer.add_image(processed, name="Processed")
    
    # Store the current parameters as attributes on the function
    update_preprocessing.current_z = example_z
    update_preprocessing.current_channel = channel
    update_preprocessing.current_preprocessed = processed
# Create parameter widgets dynamically based on selected processes
param_widgets = {}

# Now, fix the parameter widgets creation:
for process in selected_processes:
    # Get parameters for this process
    process_params = params_df[params_df['Process'] == process]
    
    for _, param in process_params.iterrows():
        param_name = param['Parameter']
        param_min = param['Min']
        param_max = param['Max']
        param_value = param['Value']
        data_type = param['Data type']
        default_value = param['Default Value']
        must_be_odd = param['Must be odd']
        param_step = (param_max - param_min) / 100
        
        # Initialize in our parameter dict
        ui_param_values[param_name] = param_value if not pd.isna(param_value) else default_value
        
        # Create widget for this parameter
        if data_type == 'Integer':
            param_step = max(1, int(param_step))  # Ensure step is at least 1 for integers
            
            # Create a properly defined widget function with the correct magicgui decorator
            widget_config = {
                "value": {"label": param_name, "min": param_min, "max": param_max, "step": param_step}
            }
            
            @magicgui(**widget_config)
            def int_param_widget(value: int = int(default_value)):
                nonlocal param_name, must_be_odd
                # If must be odd, ensure value is odd
                if must_be_odd and value % 2 == 0:
                    value += 1
                ui_param_values[param_name] = value
                # Run preprocessing to update view
                update_preprocessing()
            
            # Store the widget with a unique name
            param_widgets[param_name] = int_param_widget

        else:  # Float
            widget_config = {
                "value": {"label": param_name, "min": param_min, "max": param_max, "step": param_step}
            }
            
            @magicgui(**widget_config)
            def float_param_widget(value: float = float(default_value)):
                nonlocal param_name
                ui_param_values[param_name] = value
                # Run preprocessing to update view
                update_preprocessing()
            
            param_widgets[param_name] = float_param_widget

# Add parameter widgets (uncomment this section to actually add the widgets to the viewer)
for param_name, widget in param_widgets.items():
    viewer.window.add_dock_widget(widget, name=param_name)

@magicgui(call_button="Save Parameters")
def save_params():
    # Update parameters in the DataFrame
    for param_name, value in ui_param_values.items():
        # Find the parameter in the DataFrame
        mask = params_df['Parameter'] == param_name
        if any(mask):
            params_df.loc[mask, 'Value'] = value
    
    # Save parameters to CSV
    params_csv = os.path.join(os.path.dirname(input_processes_csv), 'parameters.csv')
    params_df.to_csv(params_csv, index=False)
    
    print(f"Parameters saved to {params_csv}")

# Add UI panels
viewer.window.add_dock_widget(update_preprocessing, name="Preprocessing")

# Add parameter widgets
#for param_name, widget in param_widgets.items():
 #   viewer.window.add_dock_widget(widget, name=param_name)

#Save widgets
viewer.window.add_dock_widget(save_params, name="Save Parameters")

# Run preprocessing once to initialize
update_preprocessing()

napari.run()