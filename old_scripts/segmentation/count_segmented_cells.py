import logging
import numpy as np
import tifffile
from cellpose import models
import napari

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting image processing...")

# Define ranges for each color as (min, max) tuples
color_ranges = {
    'red': ((50, 0, 0), (255, 80, 80)),
    'blue': ((0, 0, 50), (49, 49, 255)),
    'green': ((0, 50, 0), (49, 255, 49)),
    'purple': ((50, 0, 50), (255, 49, 255))  # Example values, adjust if needed
}

# Define representative colors for each range
representative_colors = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'purple': (128, 0, 128)  # Example values for representative colors
}

def most_common_color(colors):
    """
    Finds the most common color in an array of RGB values.

    Parameters:
    - colors: An array of RGB color values.

    Returns:
    - The most common RGB color as a tuple.
    """
    if len(colors) == 0:
        return (0, 0, 0)  # Default color if no colors are present
    # Use bincount to find the most common color
    colors_rounded = np.round(colors).astype(int)
    flat_colors = colors_rounded.view(colors_rounded.dtype.descr * colors_rounded.shape[-1])
    unique_colors, counts = np.unique(flat_colors, return_counts=True)
    most_common = unique_colors[counts.argmax()]
    return tuple(most_common)

def count_cells_by_color_with_ranges(color_stack, segmented_stack, output_image_path):
    cells_counted = 0
    """
    Counts the number of segmented cells by their color within specified ranges in a z-stack TIFF image
    and creates a new colored z-stack image.

    Parameters:
    - color_image_path: Path to the z-stack TIFF image where cells are colored by categories.
    - segmented_image_path: Path to the z-stack TIFF image where cells are segmented.
    - output_image_path: Path to save the newly created colored z-stack TIFF image.

    Returns:
    - A dictionary with color names as keys and counts as values.
    """

    # Initialize a dictionary to count cells by color
    color_counts = {color_name: 0 for color_name in color_ranges.keys()}

    # Create an output stack with the same shape as the input
    output_stack = np.zeros_like(color_stack)

    # Iterate through each slice in the z-stack
    num_slices = segmented_stack.shape[0]
    for i in range(num_slices):
        color_image = color_stack[i]
        segmented_image = segmented_stack[i]

        # Convert color image to RGB if necessary
        if color_image.ndim == 2:
            color_image_rgb = np.stack([color_image] * 3, axis=-1)
        else:
            color_image_rgb = color_image

        # Prepare the output slice
        output_slice = np.zeros_like(color_image_rgb)

        # Process each unique label in the segmented image
        unique_labels = np.unique(segmented_image)
        for label in unique_labels:
            cells_counted += 1
            if label == 0:  # Assuming 0 is the background or not a valid cell
                continue

            # Create a mask for the current cell
            cell_mask = (segmented_image == label)

            # Get all colors of the cell
            cell_colors = color_image_rgb[cell_mask]

            # Find the most common color in the cell
            cell_color_tuple = most_common_color(cell_colors)

            # Determine which color range this most common color falls into
            color_found = False
            for color_name, (min_range, max_range) in color_ranges.items():
                if all(min_range[j] <= cell_color_tuple[j] <= max_range[j] for j in range(3)):
                    color_counts[color_name] += 1
                    color_found = True
                    # Assign the representative color to the output image
                    output_slice[cell_mask] = representative_colors[color_name]
                    break

            if not color_found:
                # Optionally handle unmapped colors with a default color
                output_slice[cell_mask] = (255, 255, 255)  # White for unmapped colors

        # Add the processed slice to the output stack
        output_stack[i] = output_slice

    print(f"Total cells counted: {cells_counted}")
    # Save the new colored z-stack image
    tifffile.imwrite(output_image_path, output_stack)
    return color_counts

# Paths to the TIFF images
color_image_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/categorized_stack_pixels_semi_cropped.tiff'
segmented_image_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/cellpose_segmented_stitching_semi_cropped.tiff'
output_image_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/colored_cells_cellpose_semi_cropped.tiff'


with tifffile.TiffFile(color_image_path) as color_tiff:
    color_stack = color_tiff.asarray()

with tifffile.TiffFile(segmented_image_path) as segmented_tiff:
    segmented_stack = segmented_tiff.asarray()


# Count cells by color and create a new image
color_counts = count_cells_by_color_with_ranges(color_stack, segmented_stack, output_image_path)

# Print results
print("Cell counts by color:")
for color_name, count in color_counts.items():
    print(f"{color_name}: {count}")


# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer = napari.Viewer()
viewer.add_image(color_stack, name='output_stack')
viewer.add_labels(segmented_stack, name='Stitched Segmentation Masks')
napari.run()

