print("Running")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import tifffile
import os
import napari

viewer = napari.Viewer()

channel_names = ["Sox1", "Bra", "DAPI", "OPP"]

lower_black_threshold = 60

input_folder = '/Users/oskar/Desktop/translation_NMPs_image_analysis'
output_pathway = '/Users/oskar/Desktop/translation_NMPs_image_analysis/results_data.csv'

#image = tifffile.imread('/Users/oskar/Desktop/exp024gomb_SBSO_OPP_d5/raw_images/NM_SBSE_OPP_d5_2024_10_28__17_40_50__p4.tiff')

def remove_saturated_pixels(image, total_z, y_pixels, x_pixels, total_channels, filename):
    desaturated_image = np.zeros_like(image)
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

                if channel_2_value > blue_saturation_threshold or ((relative_channel_1 > relative_red_saturation_threshold or channel_1_value > red_saturation_threshold) and (relative_channel_0 > relative_green_saturation_threshold or channel_0_value > green_saturation_threshold)) or channel_3_value > OPP_saturation_threshold or relative_channel_3 > relative_OPP_saturation_threshold:
                    for channel in range(total_channels):
                        z_slice[channel][y,x] = 0
    
        if filename == 'NM_SBSE_OPP_d5_2024_10_28__17_40_50__p6_cropped.tiff':
            for x in range(70, 111):  # x range 70 to 110 inclusive
                for y in range(135, 161):  # y range 135 to 160 inclusive
                    # Set non-saturated pixels to zero within this region
                        for channel in range(total_channels):
                            z_slice[channel][y, x] = 0  # Set pixel to zero across all channels

        if filename == 'NM_SBSE_OPP_d5_2024_10_28__17_40_50__p7_cropped.tiff':
            for x in range(125, 190):  # x range 70 to 110 inclusive
                for y in range(64, 101):  # y range 135 to 160 inclusive
                    # Set non-saturated pixels to zero within this region
                        for channel in range(total_channels):
                            z_slice[channel][y, x] = 0  # Set pixel to zero across all channels
            for x in range(163, 183):
                for y in range(101, 118):
                   for channel in range(total_channels):
                        z_slice[channel][y, x] = 0  # Set pixel to zero across all channels 


        if filename == 'NM_SBSE_OPP_d5_2024_10_28__17_40_50__p1.tiff':
            for x in range(227, 273):  # x range 70 to 110 inclusive
                for y in range(290, 305):  # y range 135 to 160 inclusive
                    # Set non-saturated pixels to zero within this region
                        for channel in range(total_channels):
                            z_slice[channel][y, x] = 0  # Set pixel to zero across all channels


        desaturated_image[z] = z_slice



    return desaturated_image

def preprocess(image, total_z, y_pixels, x_pixels, total_channels):
    equalised_image = np.zeros_like(image)

    for z in range(total_z):
        z_slice = image[z]
        equalised_slice = np.zeros_like(z_slice)
        #desaturated_slice = remove_saturated_pixels(z_slice, y_pixels, x_pixels, total_channels)

        for channel in range(total_channels):
            channel_slice = z_slice[channel]

            # Apply adaptive histogram equalization
            equalised_slice[channel] = img_as_ubyte(equalize_adapthist(channel_slice, kernel_size=13, clip_limit=0.01, nbins=256))
        equalised_image[z] = equalised_slice

    return equalised_image

def determine_dynamic_thresholds(image, total_z, y_pixels, x_pixels, total_channels):
    average_channel_per_slice = np.zeros((total_z, total_channels))
    for channel in range(total_channels):
        channel_image = image[:, channel, :, :]
        for z in range(total_z):
            running_average = [0,0]
            channel_slice = channel_image[z]
            for x in range(x_pixels):
                for y in range(y_pixels):
                    channel_2_value = image[z, 2, y, x]
                    if channel_2_value > lower_black_threshold:
                        value = channel_slice[y, x]
                        relative_value = value / channel_2_value
                        running_average[0] += relative_value
                        running_average[1] += 1

            if running_average[1] > 0:
                average_value_for_slice = running_average[0] / running_average[1]
                average_channel_per_slice[z, channel] = average_value_for_slice

    #plt.figure(figsize=(10, 6))

    # Loop through each channel

    threshold_lines = [0, 0, 0, 0]

    for channel in range(total_channels):
        # Get values for the current channel across z-slices
        values = average_channel_per_slice[:, channel]

        # Create a sequence for z-slices
        z_slices = np.arange(total_z)
        
        # Plot the values for the current channel
        #plt.plot(z_slices, values, 'o', label=f'Channel {channel + 1}')  # 'o' for data points

        # Fit a line to the data points
        fit_coeffs = np.polyfit(z_slices, values, 4)
        fit_line = np.poly1d(fit_coeffs)


        threshold_lines[channel] = fit_line
        
        # Plot the line of best fit
        #plt.plot(z_slices, fit_line(z_slices), label=f'Best Fit for Channel {channel + 1}')

    # Labeling the plot
    '''
    plt.xlabel('Z Slice')
    plt.ylabel('Average Value')
    plt.title('Channel Values Across Z Slices with Line of Best Fit')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)  # Optional: add grid for readability
    '''
    # Show the plot
    #plt.show()

    return threshold_lines


def categorise_pixels(channel_images, z, y_pixels, x_pixels, threshold_lines):
    categorised_image = np.zeros((y_pixels, x_pixels, 3), dtype=np.uint8)  # 3 channels for RGB
    OPP_image = np.zeros((y_pixels, x_pixels), dtype=np.uint8)
    category_counts = {'Black': [], 'Orange': [], 'Red': [], 'Green': [], 'Blue': []}

    green_threshold = 45 #was 0
    relative_green_threshold = 0.47

    red_threshold = 60 #was 0
    relative_red_threshold = 0.55

    OPP_threshold = 0.55

    threshold_channel_0 = 0.8*threshold_lines[0](z) + 0.05 #was 1 and 0.23
    threshold_channel_1 = 0.85*threshold_lines[1](z) + 0. # was 0.1
    threshold_channel_3 = threshold_lines[3](z) #was 0

    low_threshold_channel_0 = 0.5*threshold_lines[0](z)
    low_threshold_channel_1 = 0.55*threshold_lines[1](z)
    
    if threshold_channel_0 > 0.1 and threshold_channel_1 > 0.1 and threshold_channel_3 > 0.1:

        for x in range(x_pixels):
            for y in range(y_pixels):
                # Get the pixel values for the three channels
                channel_0_value = channel_images[0][y, x]
                channel_1_value = channel_images[1][y, x]
                channel_2_value = channel_images[2][y, x]
                channel_3_value = channel_images[3][y, x]
                relative_channel_1 = channel_1_value / (channel_2_value + 0.001)
                relative_channel_0 = channel_0_value / (channel_2_value + 0.001)
                relative_channel_3 = channel_3_value / (channel_2_value + 0.001)
                relative_channel_3_to_threshold = relative_channel_3 / threshold_channel_3

                if relative_channel_3 > OPP_threshold:
                    if channel_2_value > lower_black_threshold:
                        OPP_image[y,x] = (150)
                        OPP_true = 1

                if channel_2_value < lower_black_threshold:
                    # Black for pixels where channel 0 value is less than the threshold
                    categorised_image[y, x] = (0, 0, 0)
                    category_counts['Black'].append([relative_channel_0, relative_channel_1, channel_2_value, relative_channel_3_to_threshold])
                    #print(relative_channel_0, relative_channel_1, relative_channel_3)
                elif relative_channel_1 > threshold_channel_1 and channel_1_value > red_threshold:
                    # Red for pixels where channel 1 is higher than threshold
                    if relative_channel_0 > threshold_channel_0 and channel_0_value > green_threshold:
                        categorised_image[y, x] = (255, 100, 25)  # NMP (overlaps)
                        category_counts['Orange'].append([relative_channel_0, relative_channel_1, channel_2_value,  relative_channel_3_to_threshold])
                    else:
                        categorised_image[y, x] = (190, 0, 0)
                        category_counts['Red'].append([relative_channel_0, relative_channel_1, channel_2_value,  relative_channel_3_to_threshold])
                elif relative_channel_0 > threshold_channel_0 and channel_1_value > green_threshold:
                    # Green for pixels where channel 2 is higher than threshold
                    categorised_image[y, x] = (50, 255, 50)
                    category_counts['Green'].append([relative_channel_0, relative_channel_1, channel_2_value,  relative_channel_3_to_threshold])
                elif relative_channel_0 < low_threshold_channel_0 and relative_channel_1 < low_threshold_channel_1:
                    # Blue for remaining pixels
                    categorised_image[y, x] = (0, 0, 255)
                    category_counts['Blue'].append([relative_channel_0, relative_channel_1, channel_2_value,  relative_channel_3_to_threshold])
                else:
                    categorised_image[y, x] = (0, 0, 0)
                    category_counts['Black'].append([relative_channel_0, relative_channel_1, channel_2_value, relative_channel_3_to_threshold])

    return category_counts, categorised_image, OPP_image

def call_categorise_pixels(image, total_z, y_pixels, x_pixels, total_channels, threshold_lines):
    accumulated_counts = {'Black': [], 'Orange': [], 'Red': [], 'Green': [], 'Blue': []}
    categorised_stack = np.zeros((total_z, y_pixels, x_pixels, 3), dtype=np.uint8)  # 3 channels for RGB
    OPP_stack = np.zeros((total_z, y_pixels, x_pixels), dtype=np.uint8)
    for z in range(total_z):
        category_counts, categorised_image, OPP_image = categorise_pixels(image[z], z, y_pixels, x_pixels, threshold_lines)

        # Add the categorized image to the stack
        categorised_stack[z] = categorised_image
        OPP_stack[z] = OPP_image

        # Accumulate counts for the DataFrame
        for category in accumulated_counts:
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

    return accumulated_counts, average_pixel_values, results, categorised_stack, OPP_stack


def process_folder(input_folder):
    results = []

    # Iterate over each file in the input directory
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if it's an image file (adjust as needed for your format, e.g., '.tif', '.png')
        if filename.endswith(('.tif', '.png', '.jpg', '.tiff')) and os.path.isfile(input_path):
            print(f"Processing {filename}...")

            # Load the image
            image = tifffile.imread(input_path)

            total_z = image.shape[0]
            total_channels = image.shape[1]
            y_pixels = image.shape[2]
            x_pixels = image.shape[3]

            desaturated_image = remove_saturated_pixels(image, total_z, y_pixels, x_pixels, total_channels, filename)
            preprocessed_image = preprocess(desaturated_image, total_z, y_pixels, x_pixels, total_channels)
        
            threshold_lines = determine_dynamic_thresholds(preprocessed_image, total_z, y_pixels, x_pixels, total_channels)

            accumulated_counts, average_pixel_values, results_values, categorised_image, OPP_image = call_categorise_pixels(preprocessed_image, total_z, y_pixels, x_pixels, total_channels, threshold_lines)
            print("\nAverage Pixel Values for Each Category:")
            print(results_values)
            #print(accumulated_counts)
            print(average_pixel_values)

            viewer.add_image(categorised_image)

            # Store only the OPP row for plotting
            opp_row = results_values.loc["OPP", ["Orange", "Red", "Green", "Blue"]]
            results.append(opp_row)

    # Combine results into a DataFrame for plotting
    results_df = pd.DataFrame(results)
    results_df.reset_index(drop=True, inplace=True)  # Reset index for a clean DataFrame
    return results_df

results_df = process_folder(input_folder)


# Rename columns with the new category names
results_df.columns = ['Sox1+ Bra+', 'Bra+', 'Sox1+', 'Other']

# Reorder columns to match the specified x-axis order
results_df = results_df[['Other', 'Bra+', 'Sox1+', 'Sox1+ Bra+']]

# Define colors for each category, with Grey for "Other" and Orange for "Sox1+ Bra+"
colors = {'Sox1+ Bra+': 'orange', 'Bra+': 'red', 'Sox1+': 'green', 'Other': 'grey', 'Background': 'black'}

print(results_df)

results_df.to_csv(output_pathway, index=False)


# Calculate mean and standard deviation for each category
means = results_df.mean()
std_devs = results_df.std()

# Plotting
plt.figure(figsize=(6, 6))

# Plot individual data points with category-specific colors
for category in results_df.columns:
    plt.scatter(
        [category] * len(results_df),
        results_df[category],
        alpha=1.0,
        s=40,
        color=colors.get(category, 'blue'),  # Ensure color consistency with the colors dictionary
        label=category if category not in plt.gca().get_legend_handles_labels()[1] else ""
    )

# Plot means with error bars
plt.errorbar(
    results_df.columns, 
    means, 
    yerr=std_devs, 
    fmt='s', 
    color='black', 
    alpha=0.4,
    capsize=5,
    markersize = 10
)

# Labels, title, and legend
plt.ylabel('Average translation intensity (Perhaps change so divide by DAPI level again? Or dont divide again)')

# Add a faint grid with y-axis spacing of 0.1 and x-axis along each category
plt.grid(True, which='both', axis='both', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # Assumes each category is at integer positions

plt.show()

napari.run()