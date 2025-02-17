def import_and_configure():
    import logging
    import numpy as np
    import tifffile
    from skimage.exposure import equalize_adapthist
    from skimage.filters import gaussian
    from skimage import img_as_ubyte
    from cellpose import models
    import cv2
    import napari
    import skimage.transform
    from scipy.ndimage import gaussian_filter
    from cellpose import denoise, io
    import pandas as pd
    import skimage.transform
    from scipy.ndimage import zoom

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting image processing...")

scale_factor = 4
z_scale_factor = 5
cell_diameter = 4.2
cropping_degree = 'very_cropped'

channel_to_segment = 2

iou_threshold = 0.5 #Stitching threshold given to be classed as the same label
cliplimit = 0.021
equalise_kernel = 31

viewer = napari.Viewer()

#Load file_pathways
logging.info("Loading file pathways...")
file_pathways = pd.read_csv('/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/file_pathways.csv')
file_pathways = file_pathways.set_index(file_pathways.columns[0])
image_pathway = (file_pathways.loc[cropping_degree, 'original'])


def resize_image(image, scale_factor):
    logging.info("Resizing...")

    # Calculate new shape
    total_z, y_pixels, x_pixels = image.shape
    new_shape = (total_z, y_pixels * scale_factor, x_pixels * scale_factor)
    
    # Resize image
    resized_image = np.zeros(new_shape, dtype=np.float32)
    
    for z in range(total_z):
        resized_image[z] = skimage.transform.resize(image[z], (y_pixels * scale_factor, x_pixels * scale_factor), anti_aliasing=True)

    resized_normalized = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min())
    image_uint8 = (resized_normalized * 255).astype(np.uint8)
    resized_image = image_uint8
    
##    print("resized", resized_image.dtype)
    
    return resized_image


def resize_all_dimensions(image, scale_factor):
    logging.info("Resizing...")

    total_z, total_channels, y_pixels, x_pixels = image.shape
    new_shape = (total_z*z_scale_factor, y_pixels * scale_factor, x_pixels * scale_factor)

    resized_image= np.zeros((total_z*z_scale_factor, total_channels, y_pixels * scale_factor, x_pixels * scale_factor), dtype=np.float32)

    # Calculate new shape
    for channel in range(total_channels):
        resizing_channel = image[:, channel, :, :]
        
        # Resize image
        resized_channel = np.zeros(new_shape, dtype=np.float32)
        
        resized_channel = skimage.transform.resize(resizing_channel, new_shape, anti_aliasing=True)

        resized_normalized = (resized_channel - resized_channel.min()) / (resized_channel.max() - resized_channel.min())
        image_uint8 = (resized_normalized * 255).astype(np.uint8)
        resized_channel = image_uint8
        resized_image[:, channel, :, :] = resized_channel
        
    return resized_image


def downsize_image(image, scale_factor):
    logging.info("Downsizing...")

    # Calculate new shape
    total_z, y_pixels, x_pixels = image.shape
    new_shape = (total_z, y_pixels // scale_factor, x_pixels // scale_factor)
    
    # Resize image
    downscaled_image = np.zeros(new_shape, dtype=np.float32)
    
    for z in range(total_z):
        downscaled_image[z] = skimage.transform.resize(image[z], (y_pixels // scale_factor, x_pixels // scale_factor), anti_aliasing=True)
    
    return downscaled_image


def cellpose_processing(image):    
    model = denoise.CellposeDenoiseModel(gpu=False, model_type="cyto3",
                                     restore_type="denoise_cyto3")

    masks, flows, styles, imgs_dn = model.eval(image, diameter=cell_diameter*scale_factor, channels=[0,0])

    print(len(np.unique(masks)))
    viewer.add_image(image)
    viewer.add_labels(masks)

def cellpose_upsample(image):
    logging.info("upsampling")
    model = denoise.DenoiseModel(gpu=False, model_type="upsample_nuclei")
    sample_output = np.squeeze(model.eval(image, diameter=cell_diameter*scale_factor))
    upsampled_y, upsampled_x = sample_output.shape
    upsampled = np.zeros((total_z, upsampled_y, upsampled_x), dtype=np.float32)

    for z in range(total_z):
        upsampled[z] = np.squeeze(model.eval(image, diameter=cell_diameter))

    print(upsampled)
    print(upsampled.shape)
    return upsampled
    
def cellpose_denoise(image):
    logging.info("Denoising...")

    denoised = np.zeros_like(image, dtype=np.float32)
    image = image.astype(np.float32)  # Convert to float32
    image /= 255.0  # Normalize to range [0, 1] if original values were [0, 255]
    for z in range(total_z):
        image_slice = image[z]
        model = denoise.DenoiseModel(gpu=False, model_type="denoise_nuclei")
        denoised[z] = np.squeeze(model.eval(image_slice, diameter=cell_diameter*scale_factor))

    return denoised
        
def cellpose_deblurr(image):
    logging.info("Deblurring...")

    deblurred = np.zeros_like(image, dtype=np.float32)
    image = image.astype(np.float32)  # Convert to float32
    image /= 255.0  # Normalize to range [0, 1] if original values were [0, 255]
    for z in range(total_z):
        image_slice = image[z]
        model = denoise.DenoiseModel(gpu=False, model_type="deblur_nuclei")
        deblurred[z] = np.squeeze(model.eval(image_slice, diameter=cell_diameter*scale_factor))

    return deblurred


def contrast_stretch(image):
    logging.info("Contrast stretching...")
    # Perform linear contrast stretching
    stretched_image = np.zeros_like(image)

    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])
        stretched_image[z] = cv2.normalize(z_slice, None, 0, 255, cv2.NORM_MINMAX)

    #print("contrast", stretched_image.dtype)
    return stretched_image

def unsharp_mask(image):
    logging.info("Unsharp masking...")

    #Unsharp masking
    unsharp_masked = np.zeros_like(image)

    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])
        # Apply Gaussian blur to the slice
        blurred = cv2.GaussianBlur(z_slice, (5, 5), sigmaX=1.0)

        # Perform unsharp masking
        unsharp_masked[z] = cv2.addWeighted(z_slice, 1.5, blurred, -0.5, 0)

    return unsharp_masked

def apply_bilateral_filter(image):
    logging.info("Applying bilateral filter...")

    #Bilateral filter
    
    bilateral_filtered = np.zeros_like(image)

    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])
        bilateral_filtered[z] = cv2.bilateralFilter(z_slice, d=9, sigmaColor=75, sigmaSpace=75)
        
    return bilateral_filtered

def sharpen_method_one(image):
    logging.info("Sharpening...")

    # Define a sharpening kernel
    sharpened_image = np.zeros_like(image)

    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])

        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        
        sharpened_image[z] = cv2.filter2D(z_slice, -1, kernel)

    return sharpened_image

def sobel_edges(image):
    # Create an empty array to store the Sobel-processed images
    sobel_image = np.zeros_like(image, dtype=np.uint8)
        
    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])

        # Apply the Sobel operator in both x and y directions
        sobelx = cv2.Sobel(z_slice, cv2.CV_64F, 1, 0, ksize=7)  # Sobel X
        sobely = cv2.Sobel(z_slice, cv2.CV_64F, 0, 1, ksize=7)  # Sobel Y
        
        # Compute the magnitude of the gradient
        sobel = cv2.magnitude(sobelx, sobely)
        
        # Normalize and convert to uint8
        sobel_image[z] = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

        #print("sobel", sobel_image.dtype)

    return sobel_image

def denoise(image):
    logging.info("Denoising...")
    denoised_image = np.zeros_like(image)
    for z in range(total_z):
        z_slice = img_as_ubyte(image[z])
        denoised_image[z] = cv2.fastNlMeansDenoising(z_slice, None, h=5, templateWindowSize=7, searchWindowSize=21)

    return denoised_image

def gaussian_filter(image):
    logging.info("Gaussian blurring...")

    # Create an empty array to store the filtered slices
    gaussian_stack = np.zeros_like(image, dtype=np.uint8)

    #print("gaussian input", image)
    
    # Iterate through each Z slice and apply the Gaussian filter
    for z in range(total_z):
        z_slice = image[z]

        # Ensure the slice is in uint8 format
        z_slice_uint8 = cv2.convertScaleAbs(z_slice)
        blurred = cv2.GaussianBlur(z_slice_uint8, (3, 3), 0)
        #blurred_twice = cv2.GaussianBlur(blurred, (3, 5), 0)
        #blurred_thrice = cv2.GaussianBlur(z_slice_uint8, (9, 5), 0)
        #print("blurred 3 times")

        gaussian_stack[z] = blurred

        #print("gaussian output", blurred)

        
    return gaussian_stack   

def equalise(image):
    logging.info("Equalising...")

    equalized_stack = np.zeros_like(image, dtype=np.uint8)
    
    for z in range(total_z):
        z_slice = image[z]

        z_slice_normalized = (z_slice - z_slice.min()) / (z_slice.max() - z_slice.min())

        equalized = img_as_ubyte(equalize_adapthist(z_slice_normalized, kernel_size=(equalise_kernel,equalise_kernel), clip_limit=cliplimit, nbins=256))
        equalized_stack[z] = equalized
    print(equalized_stack.shape)

    return equalized_stack

def sharpen_method_two(image):
    sharpened_stack = np.zeros((total_z, y_pixels, x_pixels), dtype=np.uint8)

    for z in range(total_z):
        z_slice = image[z]

        blurred = cv2.GaussianBlur(z_slice, (3, 3), 0)
        mask = cv2.subtract(z_slice, blurred)
        mask = cv2.multiply(mask, 1.5)
        sharpened = cv2.add(z_slice, mask)

        sharpened_stack[z] = sharpened

    return sharpened_stack

def threshold(image):
    block_size = 15
    C = 25 #increase for stricter thresholding
    thresholded = np.zeros_like(image, dtype = np.uint8)
    for z in range(total_z):
        z_slice = image[z]

        # Compute the local mean using a Gaussian filter
        local_mean = gaussian(z_slice, sigma=block_size)
        print(local_mean)
        
        # Subtract C from the local mean to set the threshold
        threshold = local_mean*255 - C

        print(threshold)
        
        # Apply thresholding, but retain the original pixel values
        thresholded[z] = np.where(z_slice >= threshold, z_slice, z_slice // 2)

    return thresholded

def contrast_enhance(image):
    block_size = 15
    C = 25 #increase for stricter thresholding
    contrast_enhanced = np.zeros_like(image, dtype = np.uint8)
    for z in range(total_z):
        z_slice = image[z]

        # Compute the local mean using a Gaussian filter
        local_mean = gaussian(z_slice, sigma=block_size)
        print(local_mean)
        
        # Subtract C from the local mean to set the threshold
        threshold = local_mean*255 - C

        print(threshold)
        
        # Apply thresholding, but retain the original pixel values
        thresholded[z] = np.where(z_slice >= threshold, z_slice, z_slice // 2)

    return thresholded

def overlay(image, sobel_image):
    overlayed = np.zeros_like(image)
    for z in range(total_z):
        z_slice = image[z]
        sobel_z_slice = sobel_image[z]
        #print(z_slice.dtype)
        #print(sobel_z_slice.dtype)
        overlayed[z] = cv2.addWeighted(z_slice, 1.0, sobel_z_slice, 1, 0)

    return overlayed

tifffile.imwrite((file_pathways.loc[cropping_degree, 'resized']), image)

# Use Napari for visualization
logging.info("Visualizing results with Napari...")
viewer.add_image(image)
viewer.add_image(preprocessed_image)

napari.run()
