'''
Preprocess image
Find average OPP signal across the gastruloid

'''

print("Running")


import napari.viewer
import numpy as np
import tifffile
import pandas as pd
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
import napari
import matplotlib.pyplot as plt

viewer = napari.Viewer()

image = tifffile.imread('/Users/oskar/Desktop/OPP_BS2_Oct4/cropped/actually_cropped/file_3_squeezed.tiff')

# Define channel names
channel_names = ["Oct4", "Bra", "Sox2", "OPP"]

lower_black_threshold = 110

# Obtain image properties in (slices, channels, x, y)
print(f"Image shape: {image.shape}")
total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

def remove_saturated_pixels(image):
    for z in range(total_z):
        z_slice = image[z]
        for x in range(x_pixels):
            for y in range(y_pixels):
                channel_0_value = z_slice[0][y, x]
                channel_1_value = z_slice[1][y, x]
                channel_2_value = z_slice[2][y, x]
                channel_3_value = z_slice[3][y, x]
                relative_channel_1 = channel_1_value / (channel_2_value + 0.001)
                relative_channel_0 = channel_0_value / (channel_2_value + 0.001)
                relative_channel_3 = channel_3_value / (channel_2_value + 0.001)

                blue_saturation_threshold = 255
                green_saturation_threshold = 255
                relative_green_saturation_threshold = 2.5
                red_saturation_threshold = 255
                relative_red_saturation_threshold = 2.5
                relative_OPP_saturation_threshold = 2.5
                OPP_saturation_threshold = 255

                if channel_2_value > blue_saturation_threshold or relative_channel_1 > relative_red_saturation_threshold or channel_1_value > red_saturation_threshold or relative_channel_0 > relative_green_saturation_threshold or channel_0_value > green_saturation_threshold or channel_3_value > OPP_saturation_threshold or relative_channel_3 > relative_OPP_saturation_threshold:
                    for channel in range(total_channels):
                        z_slice[channel][y,x] = 0

        if p6 == True:
            for x in range(70, 111):  # x range 70 to 110 inclusive
                for y in range(135, 161):  # y range 135 to 160 inclusive
                    # Set non-saturated pixels to zero within this region
                        for channel in range(total_channels):
                            z_slice[channel][y, x] = 0  # Set pixel to zero across all channels

        image[z] = z_slice

    return image

def preprocess(image):
    equalised_image = np.zeros_like(image)

    for z in range(total_z):
        z_slice = image[z]
        equalised_slice = np.zeros_like(z_slice)

        for channel in range(total_channels):
            channel_slice = z_slice[channel]

            # Apply adaptive histogram equalization
            equalised_slice[channel] = img_as_ubyte(equalize_adapthist(channel_slice, kernel_size=13, clip_limit=0.01, nbins=256))
        equalised_image[z] = equalised_slice

    return equalised_image

def determine_dynamic_thresholds(image):
    average_channel_per_slice = np.zeros((total_z, total_channels))
    for channel in range(total_channels):
        channel_image = image[:, channel, :, :]
        for z in range(total_z):
            running_average = [0,0]
            channel_slice = channel_image[z]
            for x in range(x_pixels):
                for y in range(y_pixels):
                    channel_3_value = image[z, 3, y, x]
                    if channel_2_value > lower_black_threshold:
                        value = channel_slice[y, x]
                        relative_value = value / channel_2_value
                        running_average[0] += relative_value
                        running_average[1] += 1

            if running_average[1] > 0:
                average_value_for_slice = running_average[0] / running_average[1]
                average_channel_per_slice[z, channel] = average_value_for_slice

    plt.figure(figsize=(10, 6))

    # Loop through each channel

    threshold_lines = [0, 0, 0, 0]

    for channel in range(total_channels):
        # Get values for the current channel across z-slices
        values = average_channel_per_slice[:, channel]

        # Create a sequence for z-slices
        z_slices = np.arange(total_z)
        
        # Plot the values for the current channel
        plt.plot(z_slices, values, 'o', label=f'Channel {channel + 1}')  # 'o' for data points

        # Fit a line to the data points
        fit_coeffs = np.polyfit(z_slices, values, 4)
        fit_line = np.poly1d(fit_coeffs)


        threshold_lines[channel] = fit_line
        
        # Plot the line of best fit
        plt.plot(z_slices, fit_line(z_slices), label=f'Best Fit for Channel {channel + 1}')

    # Labeling the plot
    plt.xlabel('Z Slice')
    plt.ylabel('Average Value')
    plt.title('Channel Values Across Z Slices with Line of Best Fit')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)  # Optional: add grid for readability

    # Show the plot
    #plt.show()

    return threshold_lines


#threshold_lines = determine_dynamic_thresholds(image)
#print(threshold_lines)


def categorise_pixels(channel_images, z, y_pixels, x_pixels):
    categorised_image = np.zeros((y_pixels, x_pixels, 3), dtype=np.uint8)  # 3 channels for RGB
    OPP_image = np.zeros((y_pixels, x_pixels), dtype=np.uint8)
    category_counts = {'Black': [], 'Blue': [], 'Yellow': [], 'Red': [], 'Green': []}

    #Black is no gastruloid, Red is rest of gastruloid, Yellow is Sox2, Blue is Oct4, Green is both

    threshold_channel_0 = 200   #Blue, Oct4
    threshold_channel_1 = 120   #Red, Bra
    threshold_channel_2 = 200   #Yellow, Sox2

    OPP_threshold = 0.55

    #threshold_channel_0 = 50000 * threshold_lines[0](z) + 0.05
    #threshold_channel_1 = 50000 * threshold_lines[1](z) + 0.0
    #threshold_channel_2 = 500 * threshold_lines[2](z)    

    if threshold_channel_0 > 0.1 and threshold_channel_1 > 0.1 and threshold_channel_2 > 0.1:
        for x in range(x_pixels):
            for y in range(y_pixels):
                # Get the pixel values for the three channels
                channel_0_value = channel_images[0][y, x]
                channel_1_value = channel_images[1][y, x]
                channel_2_value = channel_images[2][y, x]
                channel_3_value = channel_images[3][y, x]

                if channel_0_value <= threshold_channel_0 and channel_1_value <= threshold_channel_1 and channel_2_value <= threshold_channel_2:
                    # Black for pixels where channel 0 value is less than the threshold
                    #categorised_image[y, x] = (0, 0, 0)
                    category_counts['Black'].append([channel_0_value, channel_1_value, channel_2_value, channel_3_value])
                    #print(relative_channel_0, relative_channel_1, relative_channel_3)
                elif channel_2_value > threshold_channel_2:
                    # Red for pixels where channel 1 is higher than threshold
                    if channel_0_value > threshold_channel_0:
                        #categorised_image[y, x] = (0, 150, 0)  # NMP (overlaps)
                        category_counts['Green'].append([channel_0_value, channel_1_value, channel_2_value,  channel_3_value])
                    else:
                        #categorised_image[y, x] = (220, 220, 50)
                        category_counts['Yellow'].append([channel_0_value, channel_1_value, channel_2_value,  channel_3_value])
                elif channel_0_value > threshold_channel_0:
                    # Green for pixels where channel 2 is higher than threshold
                    #categorised_image[y, x] = (50, 50, 255)
                    category_counts['Blue'].append([channel_0_value, channel_1_value, channel_2_value,  channel_3_value])
                else:
                    # Red for remaining pixels
                    #categorised_image[y, x] = (190, 0, 0)
                    category_counts['Red'].append([channel_0_value, channel_1_value, channel_2_value,  channel_3_value])

    return category_counts, categorised_image, OPP_image

def call_categorise_pixels(image):
    accumulated_counts = {'Black': [], 'Blue': [], 'Yellow': [], 'Red': [], 'Green': []}
    categorised_stack = np.zeros((total_z, y_pixels, x_pixels, 3), dtype=np.uint8)  # 3 channels for RGB
    OPP_stack = np.zeros((total_z, y_pixels, x_pixels), dtype=np.uint8)
    for z in range(total_z):
        category_counts, categorised_image, OPP_image = categorise_pixels(image[z], z, y_pixels, x_pixels)

        # Add the categorized image to the stack
        categorised_stack[z] = categorised_image
        OPP_stack[z] = OPP_image

        # Accumulate counts for the DataFrame
        for category in category_counts:
            accumulated_counts[category].extend(category_counts[category])

    # Calculate average pixel values for each category
    average_pixel_values = {}
    for category in accumulated_counts:
        values = np.array(accumulated_counts[category])
        if len(values) > 0:
            average_pixel_values[category] = values.mean(axis=0)
        else:
            average_pixel_values[category] = np.array([0, 0, 0, 0])  # In case there are no pixels for a category
    

    results = pd.DataFrame(average_pixel_values, index=channel_names)
    print(results)

    return average_pixel_values, categorised_stack, OPP_stack

#viewer.add_image(image)

#desaturated_image = remove_saturated_pixels(image)
preprocessed_image = preprocess(image)

#viewer.add_image(preprocessed_image)
#napari.run()


category_counts, categorised_image, OPP_image = call_categorise_pixels(preprocessed_image)
print("\nAverage Pixel Values for Each Category:")

#viewer.add_image(OPP_image)
#viewer.add_image(categorised_image)

#napari.run()

def plot(accumulated_counts, average_pixel_values, results):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Assume accumulated_counts and average_pixel_values from your previous code
    # Convert accumulated pixel data to DataFrame for easy plotting
    df_pixels = []
    for category, pixels in accumulated_counts.items():
        if pixels:  # Only include categories with pixels
            pixels_array = np.array(pixels)
            opp_values = pixels_array[:, 3]  # OPP values (relative_channel_3)
            category_labels = np.full(opp_values.shape, category)  # Array of category names
            df_pixels.append(pd.DataFrame({'Category': category_labels, 'OPP': opp_values}))

    # Combine data for all categories into one DataFrame
    df_all_pixels = pd.concat(df_pixels)

    # Sample 1 in every 100 pixels to reduce density
    df_sampled = df_all_pixels.sample(frac=1.0, random_state=42)  # Adjust `frac` if needed

    # Map categories to numeric positions on the x-axis
    category_map = {category: idx for idx, category in enumerate(accumulated_counts.keys())}
    df_sampled['x_pos'] = df_sampled['Category'].map(category_map)

    # Add jitter by adding a small random value to each x_pos
    jitter_strength = 0.2  # Adjust this value for more or less jitter
    df_sampled['x_pos_jittered'] = df_sampled['x_pos'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df_sampled))

    # Plot the scatter plot with jittered x-positions, limited y-axis, and reduced point size
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sampled['x_pos_jittered'], df_sampled['OPP'], alpha=0.5, s=0.01)
    plt.xticks(list(category_map.values()), list(category_map.keys()))
    plt.ylim(0.5, 1)  # Set y-axis limit
    plt.xlabel('Cell Category')
    plt.ylabel('OPP (Relative Channel 3)')
    plt.title('Scatter Plot of Pixels by Cell Category with Jittered X-axis (1 in 100 pixels)')
    plt.show()

print("\nComplete")
