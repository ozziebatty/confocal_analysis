import logging
import numpy as np
import tifffile
import napari
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import numpy as np
from numba import jit
from scipy import ndimage

# Define channel names
channel_names = ["DAPI", "Bra", "Sox2", "OPP"]
channel_thresholds = [1, 0.55, 0.45, 0.5]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting image processing...")

# Load images
logging.info("Loading images...")
image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/semi_cropped_gastruloid_z.tiff')
segmented_image = tifffile.imread('/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_semi_cropped_gastruloid_z.tiff')

# Obtain image properties
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]
total_cells =len(np.unique(segmented_image))

print(f"Image shape: {image.shape} in ZCYX")
print(f"Segmented image shape: {segmented_image.shape} in ZYX")
print(total_cells, " cells to characterise")

# Define representative colours for each range
display_colours = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'purple': (128, 0, 128)
}

def preprocess(image):
    logging.info("Preprocessing...")

    preprocessed_image = np.zeros_like(image)
    
    for z in range(total_z):
        z_slice = image[z]

        #Preprocess using adapthist
        for channel in range(total_channels):
            preprocessed_image[z][channel] = img_as_ubyte(equalize_adapthist(z_slice[channel], kernel_size=None, clip_limit=0.01, nbins=256))

    return preprocessed_image

def process_slice(z_slice_image, z_slice_segmented_image, total_channels):
    labels = np.unique(z_slice_segmented_image)
    labels = labels[labels != 0]  # Skip background
    
    result = np.zeros((labels.max() + 1, total_channels + 1), dtype=np.float64)
    
    for label in labels:
        masked_cell = (z_slice_segmented_image == label)
        pixel_count = np.sum(masked_cell)
        channel_sums = np.array([np.sum(z_slice_image[channel][masked_cell]) for channel in range(total_channels)])
        result[label] = np.concatenate(([pixel_count], channel_sums))
    
    return result

def quantify_cell_fluorescence(image, segmented_image):
    logging.info("Quantifying fluorescence...")
    
    # Process all z-slices
    cell_fluorescence = np.sum([process_slice(image[z], segmented_image[z], total_channels) for z in range(total_z)], axis=0)
    
    # Normalize channel intensities by pixel count
    pixel_counts = cell_fluorescence[:, 0]
    mask = pixel_counts > 0
    cell_fluorescence[mask, 1:] /= pixel_counts[mask, np.newaxis]
    
    # Create structured array for final result
    result = np.zeros(total_cells, dtype=[('cell_number', int), ('channels', float, total_channels)])
    result['cell_number'] = np.arange(total_cells)
    result['channels'] = cell_fluorescence[:, 1:]
    
    # Normalize intensities relative to DAPI
    logging.info("Normalising intensities to DAPI...")
    dapi_values = result['channels'][:, 0]
    mask = dapi_values > 0
    result['channels'][mask] /= dapi_values[mask, np.newaxis]
    
    return result


def characterise_cells(cell_fluorescence):
    logging.info("Characterising cells...")    
    total_Bra, total_Sox2, total_Bra_Sox2, total_unlabelled = 0, 0, 0, 0
    characterised_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)

    for cell in cell_fluorescence:
        cell_number = cell['cell_number']
        if cell_number % 100 == 0:
            print(cell_number, " cells characterised") 
        if cell_number == 0:  # Skip background
            continue
        if cell['channels'][1] > channel_thresholds[1]:
            if cell['channels'][2] > channel_thresholds[2]:
                total_Bra_Sox2 += 1
                colour = display_colours['purple']
            else:
                total_Bra += 1
                colour = display_colours['red']
        elif cell['channels'][2] > channel_thresholds[2]:
            total_Sox2 += 1
            colour = display_colours['green']
        else:
            total_unlabelled += 1
            colour = display_colours['blue']

        #characterised_image[segmented_image == cell_number] = colour

    print("Total Bra+ : ", total_Bra)
    print("Total Sox2+ : ", total_Sox2)
    print("Total Bra+ Sox2+ : ", total_Bra_Sox2)
    print("Total unlabelled : ", total_unlabelled)

    return characterised_image


preprocessed_image = preprocess(image)
normalised_cell_fluorescence = quantify_cell_fluorescence(preprocessed_image, segmented_image)
characterised_image = characterise_cells(normalised_cell_fluorescence)


#Visualise using Napari
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(characterised_image, name='Characterised image')
viewer.add_labels(segmented_image, name='Segmentation Masks')
napari.run()

