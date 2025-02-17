
from datetime import datetime
#print(f"{datetime.now():%H:%M:%S} - Importing packages...")

import napari.viewer
import numpy as np
import tifffile
import pandas as pd
import napari
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
from skimage import filters
import sys
import ast

##sampling_frequency = 1

brightness_tweak = [1.4, 1.4, 1.5, 2, 2]

#channel_0_to_2_contribution = 0.489 #Estimating around 0.1
#channel_4_to_2_contribution = 0.784 #Estimating around 0.5

#channel_0_to_1_contribution = 0.45 #Should be near 0
channel_3_to_1_contribution = 0.16645300643253003
channel_4_to_1_contribution = 0.6262448209596455
z_position_to_1_contribution = -0.009453877255236373

debug_mode = False

if debug_mode == True:
    folder_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/SBSE_p1'
    threshold_folder = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/SBSE_p1'
    cpu_num = ''
    channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "DAPI", "Bra"]
    dapi_channel = 3
    display_napari = 'False'
else:
    folder_pathway = sys.argv[1]
    #print(folder_pathway)
    threshold_folder = sys.argv[2]
    #print(threshold_folder)
    cpu_num = ''#sys.argv[3]
    channel_names = ast.literal_eval(sys.argv[4])
    dapi_channel = int(sys.argv[5])
    display_napari = sys.argv[6]


#print(f"{datetime.now():%H:%M:%S} - Loading images...")
image_pathway = folder_pathway + '/preprocessed' + cpu_num + '.tiff'
deconvoluted_image_pathway = folder_pathway + '/deconvoluted' + cpu_num + '.tiff'
image = tifffile.imread(image_pathway)

total_z = image.shape[0]
total_channels = image.shape[1]
y_pixels = image.shape[2]
x_pixels = image.shape[3]

#SBSO_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSO_example_image_1_cropped.tiff'
#SBSE_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_example_image_cropped.tiff'
#demultiplexed_SBSO_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSO_example_image_1_demultiplexed_cropped.tiff'
#demultiplexed_SBSE_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_example_image_demultiplexed_cropped.tiff'

def load_image(image_pathway):
    print(f"{datetime.now():%H:%M:%S} - Loading image...")

    image = tifffile.imread(image_pathway)

    print("Image shape (z, c, y, x) :", image.shape, "dtype:", image.dtype)
    total_z = image.shape[0]
    total_channels = image.shape[1]
    y_pixels = image.shape[2]
    x_pixels = image.shape[3]

    return image, total_z, total_channels, y_pixels, x_pixels

def preprocess(image):
    print(f"{datetime.now():%H:%M:%S} - Preprocessing...")

    def EAH(channel_slice):     
        transposed_equalised_stack = img_as_ubyte(equalize_adapthist(channel_slice, kernel_size=(equalise_kernel), clip_limit=clip_limit, nbins=n_bins))
        equalised_stack = transposed_equalised_stack.transpose()

        return equalised_stack

    def gaussian_filter(channel_slice):
        # Create an empty array to store the filtered slices
        transposed_blurred_stack = img_as_ubyte(filters.gaussian(channel_slice, sigma = blurr_level))
            
        return transposed_blurred_stack   

    equalise_kernel = (64,64,10)
    clip_limit = 0.005 #Higher is stronger
    n_bins = 256

    blurr_level = 0.55 #in st_dev

    preprocessed_image = image.copy()
    equalised_blurred_image = image.copy()

    for channel in range(total_channels):
        channel_to_preprocess = channel
        channel_slice = image[:, channel_to_preprocess, :, :]
        transposed_channel_slice = channel_slice.transpose()
        transposed_blurred_channel_slice = gaussian_filter(transposed_channel_slice)

        equalised_blurred_channel_slice = EAH(transposed_blurred_channel_slice)

        equalised_blurred_image[:, channel_to_preprocess, :, :] = equalised_blurred_channel_slice  

        #preprocessed_image[:, channel_to_preprocess, :, :] = equalised_channel_slice  

    return equalised_blurred_image

def postprocess(image):
    print(f"{datetime.now():%H:%M:%S} - Postprocessing...")

    def EAH(channel_slice):     
        transposed_equalised_stack = img_as_ubyte(equalize_adapthist(channel_slice, kernel_size=(equalise_kernel), clip_limit=clip_limit, nbins=n_bins))
        equalised_stack = transposed_equalised_stack.transpose()

        return equalised_stack

    def gaussian_filter(channel_slice):
        # Create an empty array to store the filtered slices
        transposed_blurred_stack = img_as_ubyte(filters.gaussian(channel_slice, sigma = blurr_level))
            
        return transposed_blurred_stack   

    equalise_kernel = (64,64,10)
    clip_limit = 0.002 #Higher is stronger
    n_bins = 256

    blurr_level = 0.6 #in st_dev

    equalised_blurred_image = image.copy()

    for channel in range(total_channels):
        if channel == 1 or channel == 2:    
            channel_to_preprocess = channel
            channel_slice = image[:, channel_to_preprocess, :, :]
            transposed_channel_slice = channel_slice.transpose()
            transposed_blurred_channel_slice = gaussian_filter(transposed_channel_slice)

            equalised_blurred_channel_slice = EAH(transposed_blurred_channel_slice)

            equalised_blurred_image[:, channel_to_preprocess, :, :] = equalised_blurred_channel_slice  

    return equalised_blurred_image   

def calculate_pixel_values(image):
    total_pixels = 0
    pixel_data = []
    for z in range(total_z):
        z_slice = image[z]
        print(f"{datetime.now():%H:%M:%S} - Processing slice z =", z)
        print(f"{datetime.now():%H:%M:%S} - Number of pixels = ", total_pixels)
        for x in range(x_pixels):
            if x%sampling_frequency == 0: #Speed up processing by taking only a sample of pixels
                for y in range(y_pixels):
                    if y%sampling_frequency == 0: #Speed up processing by taking only a sample of pixels
                        channel_3_value = z_slice[3][y, x]
                        if channel_3_value > 0:
                            total_pixels += 1
                            channel_0_value = z_slice[0][y, x]
                            channel_1_value = z_slice[1][y, x]
                            channel_2_value = z_slice[2][y, x]
                            channel_4_value = z_slice[4][y, x]

                            pixel_data.append({
                                "Sox1": channel_0_value,
                                "Sox2_Cyan": channel_1_value,
                                "Sox2_Orange": channel_2_value,
                                "DAPI": channel_3_value,
                                "Bra": channel_4_value,
                            })

    pixel_dataframe = pd.DataFrame(pixel_data)
    return pixel_dataframe

def calculate_average_channel_value(image, channel):
    channel_slice = np.array(image[:, channel, :, :])

    # Get the non-zero values
    non_zero_values = channel_slice[channel_slice > 0]

    # Calculate the mean of the non-zero values
    if non_zero_values.size > 0:  # Check if there are any non-zero values
        mean_non_zero = np.mean(non_zero_values)
    else:
        mean_non_zero = 0  # Or np.nan, depending on your use case

    print("Mean of non-zero values:", mean_non_zero)

    return mean_non_zero

def calculate_average_channel_z_slice_value(channel_z_slice):
    # Get the non-zero values
    non_zero_values = channel_z_slice[channel_z_slice > 0]

    # Calculate the mean of the non-zero values
    if non_zero_values.size > 0:  # Check if there are any non-zero values
        mean_non_zero = np.mean(non_zero_values)
    else:
        mean_non_zero = 0  # Or np.nan, depending on your use case

    return mean_non_zero

def deconvolute(image):
    for z in range(total_z):
        z_slice = image[z]
        if (z + 1) * 10 // total_z > z * 10 // total_z:  # Check if percentage milestone is crossed
            print(f"{datetime.now():%H:%M:%S} - Deconvolution {((z + 1) * 100) // total_z}% complete")
        for x in range(x_pixels):
            for y in range(y_pixels):
                channel_3_value = z_slice[3][y, x]
                if channel_3_value > 0:
                    channel_0_value = z_slice[0][y, x]
                    channel_1_value = z_slice[1][y, x]
                    channel_2_value = z_slice[2][y, x]
                    channel_4_value = z_slice[4][y, x]

                    '''
                    #Remove contributions of channel_0 and channel_4 from channel_2
                    channel_2_negative_offset = channel_0_to_2_contribution * channel_0_value + channel_4_to_2_contribution * channel_4_value
                    channel_2_offset_value = channel_2_value - channel_2_negative_offset
                    if channel_2_offset_value < 0:
                        channel_2_offset_value = 0
                    z_slice[2][y,x] = channel_2_offset_value

                    '''
                    #Remove contributions of channel_0 and channel_4 from channel_1
                    channel_1_negative_offset = channel_3_to_1_contribution * channel_3_value + channel_4_to_1_contribution * channel_4_value + z_position_to_1_contribution * z
                    channel_1_offset_value = channel_1_value - channel_1_negative_offset
                    if channel_1_offset_value < 0:
                        channel_1_offset_value = 0
                    z_slice[1][y,x] = channel_1_offset_value

                    '''
                    #Tweak brightness of remaining channels
                    z_slice[0][y, x] = channel_0_value * brightness_tweak[0] / relative_DAPI
                    z_slice[3][y, x] = channel_3_value * brightness_tweak[3] / relative_DAPI
                    z_slice[4][y, x] = channel_4_value * brightness_tweak[4] / relative_DAPI
                    '''
                    
    return image

def plot_channel_scatterplots(pixel_dataframe):
    print(f"{datetime.now():%H:%M:%S} - Plotting...")
    
    # Create scatter plots for each pair of channels
    for i in range(len(channels)):
        for j in range(len(channels)):
            x_channel = channels[i]
            y_channel = channels[j]


            x_data = pixel_dataframe[x_channel]
            y_data = pixel_dataframe[y_channel]
            slope, intercept = np.polyfit(x_data, y_data, 1)
            line = slope * x_data + intercept

            # Create a scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(
                pixel_dataframe[x_channel], 
                pixel_dataframe[y_channel], 
                s=1,  # Marker size
                alpha=0.5  # Transparency for better visibility
            )

            # Plot the line of best fit
            plt.plot(
                x_data, 
                line, 
                color='red', 
                linewidth=2, 
                label=f"Fit: y = {slope:.2f}x + {intercept:.2f}"
            )

            plt.title(f"{x_channel} vs {y_channel}", fontsize=14)
            plt.xlabel(x_channel, fontsize=12)
            plt.ylabel(y_channel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save or show the plot
            plt.show()

def multichannel_display(image):
    print(f"{datetime.now():%H:%M:%S} - Loading into Napari...")

    colours = ['green', 'cyan', 'orange', 'grey', 'red']
    viewer = napari.Viewer()

    print(image.shape)

    # Add each channel as a separate layer with its own colour
    for channel, colour in enumerate(colours):
        viewer.add_image(
            image[:, channel, :, :], 
            name=channels[channel],
            colormap=colour,
            blending='additive',  # Use additive blending for transparency
            opacity=0.7  # Adjust opacity as needed
        )

def find_contributions(SBSO_image, SBSE_image, SBSO_relative_DAPI, SBSE_relative_DAPI):
    viewer = napari.Viewer()

    SBSO_z_slice = SBSO_image[3] * SBSO_relative_DAPI
    SBSO_red = SBSO_z_slice[4]
    SBSO_cyan = SBSO_z_slice[1]
    SBSO_orange = SBSO_z_slice[2]


    SBSE_z_slice = SBSE_image[3] * SBSE_relative_DAPI
    SBSE_red = SBSE_z_slice[4]
    SBSE_cyan = SBSE_z_slice[1]
    SBSE_orange = SBSE_z_slice[2]

    intensity_values = []
    contribution_value = 0.4
    while contribution_value < 0.7:
        corrected_SBSO_cyan = SBSO_cyan - (contribution_value * SBSO_red)

        corrected_SBSO_orange = SBSO_orange - (contribution_value * SBSO_red)

        corrected_SBSE_cyan = SBSE_cyan - (contribution_value * SBSE_red)
        corrected_SBSE_orange = SBSE_orange - (contribution_value * SBSE_red)

        corrected_SBSO_cyan[corrected_SBSO_cyan < 0] = 0
        corrected_SBSO_orange[corrected_SBSO_orange < 0] = 0
        corrected_SBSE_cyan[corrected_SBSE_cyan < 0] = 0
        corrected_SBSE_orange[corrected_SBSE_orange < 0] = 0

        #viewer.add_image(corrected_SBSO_orange, name=f"SBSO_orange {contribution_value}")
        #viewer.add_image(corrected_SBSE_orange, name=f"SBSE_orange {contribution_value}")
        viewer.add_image(corrected_SBSO_cyan, name=f"SBSO_cyan {contribution_value}")
        viewer.add_image(corrected_SBSE_cyan, name=f"SBSE_cyan {contribution_value}")

        SBSO_cyan_intensity_value = calculate_average_channel_z_slice_value(corrected_SBSO_cyan)
        SBSO_orange_intensity_value = calculate_average_channel_z_slice_value(corrected_SBSO_orange)
        SBSE_cyan_intensity_value = calculate_average_channel_z_slice_value(corrected_SBSE_cyan)
        SBSE_orange_intensity_value = calculate_average_channel_z_slice_value(corrected_SBSE_orange)
        relative_cyan_intensity = SBSE_cyan_intensity_value / SBSO_cyan_intensity_value
        relative_orange_intensity = SBSO_orange_intensity_value / SBSE_orange_intensity_value

        # Append the values to the list as a dictionary
        intensity_values.append({
            "SBSO_cyan_intensity": SBSO_cyan_intensity_value,
            "SBSO_orange_intensity": SBSO_orange_intensity_value,
            "SBSE_cyan_intensity": SBSE_cyan_intensity_value,
            "SBSE_orange_intensity": SBSE_orange_intensity_value,
            "relative_cyan_intensity": relative_cyan_intensity,
            "relative_orange_intensity": relative_orange_intensity,
        })

        intensity_values_df = pd.DataFrame(intensity_values)

        contribution_value += 0.05
    
    print(intensity_values_df)

def find_contributions_two(threshold_channel, leaky_channel, stained_channel, image):
    leaky_channel_slice = (image[:, leaky_channel, :, :]).flatten()
    stained_channel_slice = (image[:, stained_channel, :, :]).flatten()
    threshold_channel_slice = (image[:, threshold_channel, :, :]).flatten()
    total_pixels = len(leaky_channel_slice)

    def find_overlap_coefficient(leaky_channel_slice, stained_channel_slice):
        total_overlap = np.float32(0)
        total_leaky_channel_intensity_squared = np.float32(0)
        for pixel in range(total_pixels):
            leaky_channel_intensity = np.float32(leaky_channel_slice[pixel])
            stained_channel_intensity = np.float32(stained_channel_slice[pixel])
            total_overlap += leaky_channel_intensity * stained_channel_intensity
            total_leaky_channel_intensity_squared += leaky_channel_intensity * leaky_channel_intensity
        
        #print("total overlap", total_overlap)
        #print("total leaky channel squared", total_leaky_channel_intensity_squared)

        overlap_coefficient = total_overlap / total_leaky_channel_intensity_squared

        print("Overlap coefficient", overlap_coefficient)

        return overlap_coefficient

    overlap_threshold = find_overlap_coefficient(threshold_channel_slice, stained_channel_slice) #Use DAPI overlap as a threshold
    print("Overlap threshold =", overlap_threshold)

    contribution_value = 0
    overlap_coefficient = overlap_threshold + 1
    while overlap_coefficient > overlap_threshold:
        corrected_stained_channel_slice = stained_channel_slice - contribution_value * leaky_channel_slice
        overlap_coefficient = find_overlap_coefficient(leaky_channel_slice, corrected_stained_channel_slice)
        print("Contribution value =", contribution_value, "with overlap coefficient =", overlap_coefficient)
        contribution_value += 0.01


    #Perhaps must be lower than DAPI
    print("contribution value of leaky channel", leaky_channel, " into channel", stained_channel, " = ", contribution_value)

    return contribution_value

def find_auto_fluorescent_cells(image, DAPI_channel, channel_1, channel_2, channel_3, channel_4):
    channel_slice_shape = (image[:, 0, :, :]).shape
    DAPI_channel_slice = (image[:, DAPI_channel, :, :]).flatten()
    channel_1_slice = (image[:, channel_1, :, :]).flatten()
    channel_2_slice = (image[:, channel_2, :, :]).flatten()
    channel_3_slice = (image[:, channel_3, :, :]).flatten()
    channel_4_slice = (image[:, channel_4, :, :]).flatten()
    total_pixels = len(DAPI_channel_slice)
    auto_fluorescence_slice = np.zeros(total_pixels)

    def create_auto_fluorescence_slice():
        total_fluorescence = 0
        for pixel in range(total_pixels):
            DAPI_channel_fluorescence = DAPI_channel_slice[pixel].astype(np.float64)
            if DAPI_channel_fluorescence > 0:
                channel_1_fluorescence = channel_1_slice[pixel].astype(np.float64)
                channel_2_fluorescence = channel_2_slice[pixel].astype(np.float64)
                channel_3_fluorescence = channel_3_slice[pixel].astype(np.float64)
                channel_4_fluorescence = channel_4_slice[pixel].astype(np.float64)
                total_fluorescence = DAPI_channel_fluorescence * channel_1_fluorescence * channel_2_fluorescence * channel_3_fluorescence# * channel_4_fluorescence
                #total_DAPI_fluorescence_cubed = np.power(DAPI_channel_fluorescence, 2)
            
                #auto_fluorescence_coefficient = total_fluorescence / total_DAPI_fluorescence_cubed

                auto_fluorescence_slice[pixel] = np.power(total_fluorescence, 0.265)

                #print(auto_fluorescence_coefficient)

    create_auto_fluorescence_slice()

    print(np.max(auto_fluorescence_slice))

    auto_fluorescence_slice = (auto_fluorescence_slice.reshape(channel_slice_shape)).astype(np.uint8)


    return auto_fluorescence_slice

def remove_autofluorescence(image, autofluorescence_channel):
    def amplify_autofluorescence(channel_slice):
        def gaussian_filter(channel_slice, blurr_level):
            blurred_stack = (img_as_ubyte(filters.gaussian(channel_slice, sigma = blurr_level))).astype(np.float32)
                
            return blurred_stack  

        channel_slice[channel_slice > 70 ] = 255

        small_blurred = gaussian_filter(channel_slice, 0.1)
        large_blurred = gaussian_filter(channel_slice, 5)

        difference = (small_blurred - 2*large_blurred)
        difference[difference < 0] = 0 #Ensures no negative values
        difference = difference.astype(np.uint8)

        postprocessed_difference = gaussian_filter(difference, 0.5)
        postprocessed_difference = postprocessed_difference.astype(np.uint8)

        max_value = np.max(postprocessed_difference)
        normalise_value = 255 / max_value 

        autofluorescence_slice = (postprocessed_difference * normalise_value).astype(np.uint8)

        return autofluorescence_slice

    channel_slice = image[:, autofluorescence_channel, :, :]

    original_channel_slice = channel_slice.copy().astype(np.float32)

    #viewer.add_image(original_channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    channel_slice = amplify_autofluorescence(channel_slice)
    autofluorescence_slice = amplify_autofluorescence(channel_slice)
    #viewer.add_image(autofluorescence_slice)

    #corrected_channel_slice = original_channel_slice - autofluorescence_slice
    #corrected_channel_slice[corrected_channel_slice < 0] = 0 #Ensures no negative values
    #corrected_channel_slice = corrected_channel_slice.astype(np.uint8)

    #viewer.add_image(corrected_channel_slice)

    total_channels = image.shape[1]
    corrected_image = np.zeros_like(image)

    for channel in range(total_channels):
        channel_slice = image[:, channel, :, :].astype(np.float32)
        channel_slice = channel_slice - autofluorescence_slice
        channel_slice[channel_slice < 0] = 0 #Ensures no negative values
        channel_slice = channel_slice.astype(np.uint8)

        corrected_image[:, channel, :, :] = channel_slice

    return corrected_image

deconvoluted_image = deconvolute(image)

tifffile.imwrite(deconvoluted_image_pathway, deconvoluted_image)

print("Succesfully deconvoluted")

"""
#SBSE_image, total_z, total_channels, y_pixels, x_pixels = load_image(SBSE_image_pathway)
#multichannel_display(SBSE_image)
#preprocessed_SBSE_image = preprocess(SBSE_image)
#SBSE_image_corrected = remove_autofluorescence(preprocessed_SBSE_image, 2)
#multichannel_display(SBSE_image_corrected)

#napari.run()

#find_contributions_new(3, 0, 2, SBSE_image_corrected)
#find_contributions_new(3, 4, 2, SBSE_image_corrected)


SBSO_image, total_z, total_channels, y_pixels, x_pixels = load_image(SBSO_image_pathway)
preprocessed_SBSO_image = preprocess(SBSO_image)
SBSO_image_corrected = remove_autofluorescence(preprocessed_SBSO_image, 1)

find_contributions_new(3, 0, 1, SBSO_image)
find_contributions_new(3, 4, 1, SBSO_image)

multichannel_display(SBSO_image_corrected)

multichannel_display(SBSE_image_corrected)

 
print(f"{datetime.now():%H:%M:%S} - Demultiplexing SBSE...")
SBSE_image, total_z, total_channels, y_pixels, x_pixels = load_image(SBSE_image_pathway)
SBSE_image_corrected = remove_autofluorescence(SBSE_image, 2)
preprocessed_SBSE_image = preprocess(SBSE_image_corrected)
normal_DAPI = calculate_average_channel_value(preprocessed_SBSE_image, 3)
relative_DAPI = 1
demultiplexed_SBSE = demultiplex(preprocessed_SBSE_image)
postprocessed_SBSE = postprocess(demultiplexed_SBSE)
multichannel_display(postprocessed_SBSE)
tifffile.imwrite(demultiplexed_SBSE_image_pathway, demultiplexed_SBSE)


print(f"{datetime.now():%H:%M:%S} - Demultiplexing SBSO...")
SBSO_image, total_z, total_channels, y_pixels, x_pixels = load_image(SBSO_image_pathway)
preprocessed_SBSO_image = preprocess(SBSO_image)
relative_DAPI = calculate_average_channel_value(preprocessed_SBSO_image, 3) / normal_DAPI #Should be normalised to DAPI (Average non 0 values per image)
demultiplexed_SBSO = demultiplex(SBSO_image)
postprocessed_SBSO = postprocess(demultiplexed_SBSO)
multichannel_display(postprocessed_SBSO)
tifffile.imwrite(demultiplexed_SBSO_image_pathway, demultiplexed_SBSO)

print(f"{datetime.now():%H:%M:%S} - Running Napari...")
napari.run()

"""



