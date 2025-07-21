import cv2
import numpy as np
import tifffile as tiff

# Define the exact RGB color values
colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'purple': (255, 0, 255)  # Red + Blue
}

def count_cells_by_color(image_path, output_image_path):
    # Load the image stack (TIFF file with multiple pages)
    success, images = cv2.imreadmulti(image_path, [], cv2.IMREAD_UNCHANGED)

    if not success or images is None:
        raise FileNotFoundError(f"Image not found or unable to load from {image_path}. Please check the path and file format.")

    total_cell_counts = {color: 0 for color in colors.keys()}

    # List to store all debug images
    debug_images = []

    # Iterate over each plane in the Z-stack
    for idx, image in enumerate(images):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Distance transform
        dist_transform = cv2.distanceTransform(clean_binary, cv2.DIST_L2, 5)

        # Threshold to obtain the sure foreground area
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Find the sure background area
        sure_bg = cv2.dilate(clean_binary, kernel, iterations=3)

        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region as zero
        markers[unknown == 255] = 0

        # Apply the watershed algorithm
        markers = cv2.watershed(image_rgb, markers)

        # Create an image for debugging
        debug_image = image_rgb.copy()

        # Dictionary to hold cell counts for this plane
        cell_counts = {color: 0 for color in colors.keys()}

        # Iterate over each detected region
        for marker in range(2, num_labels + 1):  # Start from 2 because 1 is the background
            # Create a mask for the current marker
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[markers == marker] = 255

            # Find the contour for this marker
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calculate the moments of the contour to find the centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                else:
                    continue

                # Get the color of the cell by looking at the center pixel
                cell_color = tuple(image_rgb[cY, cX])

                # Increment the count for the detected color
                for color_name, rgb_value in colors.items():
                    if cell_color == rgb_value:
                        cell_counts[color_name] += 1
                        # Mark the centroid in white for debugging
                        debug_image[cY, cX] = (255, 255, 255)
                        break

        # Update the total cell counts across all planes
        for color in total_cell_counts:
            total_cell_counts[color] += cell_counts[color]

        # Add the debug image to the list
        debug_images.append(debug_image)

    # Save the debug images as a single TIFF file
    tiff.imwrite(output_image_path, np.array(debug_images), photometric='rgb')
    print(f"Saved debug image stack to: {output_image_path}")

    return total_cell_counts

# Example usage
image_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/categorized_stack_pixels.tiff'
output_image_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/categorized_stack_pixels_marked.tiff'
cell_counts = count_cells_by_color(image_path, output_image_path)
print("Total cell counts by color across all planes:", cell_counts)
