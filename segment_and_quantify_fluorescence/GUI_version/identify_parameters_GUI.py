#with napari
print("Running")
import numpy as np
import tifffile
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import napari
from magicgui import magicgui
from cellpose import models
from datetime import datetime
from skimage.segmentation import find_boundaries
from cv2 import GaussianBlur


input_path = '/Users/oskar/Desktop/format_test/SBSO_stellaris.tiff'


# Load image
image = tifffile.imread(input_path)
print("Original image shape (z, c, y, x):", image.shape, "dtype:", image.dtype)
total_z = image.shape[0]
total_channels = image.shape[1]

# Napari Viewer
viewer = napari.Viewer()

# Function for preprocessing
def EAH(channel_z_slice, kernel_size, clip_limit, nbins):
    return img_as_ubyte(equalize_adapthist(channel_z_slice, kernel_size=(kernel_size, kernel_size), clip_limit=clip_limit, nbins=nbins))

def segment(channel_z_slice, cell_diameter, flow_threshold, cellprob_threshold):
    model = models.Cellpose(gpu=False, model_type='nuclei')  # Use 'cyto2' for segmentation
    segmented_channel_z_slice, flows, styles, diams = model.eval(
        channel_z_slice,
        diameter=cell_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return segmented_channel_z_slice

# Interactive Preprocessing UI
@magicgui(
    example_z={"label": "Z slice", "min": 0, "max": total_z - 1, "step": 1},
    channel={"label": "Channel", "min": 0, "max": total_channels - 1, "step": 1},
    EAH_kernel_size={"label": "Kernel size", "min": 4, "max": 50, "step": 1},
    clip_limit={"label": "Clip Limit", "min": 0, "max": 0.2, "step": 0.001},
    n_bins={"label": "N Bins", "min": 64, "max": 512, "step": 64},
    gaussian_kernel_size={"label": "Gaussian Kernel size", "min": 3, "max": 51, "step": 2},
    sigma={"label": "Sigma", "min": 0, "max": 1, "step": 0.05},
)
def update_preprocessing(example_z: int = 5, channel: int = 3, EAH_kernel_size: int = 16, clip_limit: float = 0.005, n_bins: int = 256,
                         gaussian_kernel_size: int = 11, sigma: float = 0.1):
    # Extract single-channel slice
    channel_z_slice = image[example_z, channel, :, :]
    equalised = EAH(channel_z_slice, EAH_kernel_size, clip_limit, n_bins)
    preprocessed = GaussianBlur(equalised, ksize = (gaussian_kernel_size, gaussian_kernel_size), sigmaX = sigma)

    if "Original" in viewer.layers:
        viewer.layers["Original"].data = channel_z_slice
    else:
        viewer.add_image(channel_z_slice, name="Original")
        
    if "Processed" in viewer.layers:
        viewer.layers["Processed"].data = preprocessed
    else:
        viewer.add_image(preprocessed, name="Processed")
    
    # Store the current parameters as attributes on the function
    update_preprocessing.current_z = example_z
    update_preprocessing.current_channel = channel
    update_preprocessing.current_preprocessed = preprocessed

# Interactive Segmentation UI
@magicgui(
    cell_diameter={"label": "Cell Diameter", "min": 2, "max": 50, "step": 1},
    flow_threshold={"label": "Flow Threshold", "min": -5, "max": 5, "step": 0.02},
    cellprob_threshold={"label": "Cell Prob Threshold", "min": -10, "max": 10, "step": 0.02},
)
def update_segmentation(cell_diameter: int = 8, flow_threshold: float = 0.5, 
                    cellprob_threshold: float = 0.5):
    # Get the current preprocessed image from the preprocessing function
    preprocessed = getattr(update_preprocessing, 'current_preprocessed', None)
    
    if preprocessed is None:
        print("Please run preprocessing first")
        return
    
    # Run segmentation
    segmented = segment(preprocessed, cell_diameter, flow_threshold, cellprob_threshold)
    
    # Show the full segmentation
    if "Segmented" in viewer.layers:
        viewer.layers["Segmented"].data = segmented
        viewer.layers["Segmented"].visible = True
    else:
        viewer.add_labels(segmented, name="Segmented")

@magicgui(call_button="Save Parameters")
def save_params_widget():
    print(message)
    return message

# Add UI panels
viewer.window.add_dock_widget(update_preprocessing, name="Preprocessing")
viewer.window.add_dock_widget(update_segmentation, name="Segmentation")
viewer.window.add_dock_widget(save_params_widget, name="Save Parameters")

# Run preprocessing once to initialize
update_preprocessing()
update_segmentation()

napari.run()