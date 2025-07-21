import os
import tifffile
import numpy as np

file_path = '/Users/oskar/Desktop/format_test/SBSO_stellaris/SBSO_stellaris_segmentation.tiff'

def load_image(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension in ['.tiff', '.tif', '.lsm']:
        with tifffile.TiffFile(file_path) as tif:
            image = tif.asarray()
            
            # Extract voxel size and dimension information from metadata
            voxel_size, dimensions = extract_metadata(tif, image.shape)
            
            print(f"Image shape: {image.shape}")
            print(f"Dimensions: {dimensions}")
            print(f"Voxel size: {voxel_size}")
            
            return image, voxel_size, dimensions
    else:
        raise ValueError("Unsupported file format. File must end in '.tiff', '.tif', or '.lsm'.")

def extract_metadata(tif, image_shape):
    """Extract voxel size and dimension information from metadata."""

    # For LSM files
    if hasattr(tif, 'lsm_metadata') and tif.lsm_metadata is not None:
        lsm_info = tif.lsm_metadata
        
        def find_voxel_size(tif):
            # Extract voxel size with 4 significant figures
            voxel_size = None
            try:
                voxel_size = {
                    'x': format_sig_figs(lsm_info['VoxelSizeX'] * 1e6, 4),
                    'y': format_sig_figs(lsm_info['VoxelSizeY'] * 1e6, 4),
                    'z': format_sig_figs(lsm_info['VoxelSizeZ'] * 1e6, 4)
                }
            except:
                raise ValueError("Could not extract voxel size information from file metadata.")

            return voxel_size

        def find_dimensions(tif):

            dimensions = {}
            try:
                dimensions['x'] = lsm_info['DimensionX']
                dimensions['y'] = lsm_info['DimensionY']
                dimensions['z'] = lsm_info['DimensionZ']
                dimensions['c'] = lsm_info['DimensionChannels']
                dimensions['t'] = lsm_info['DimensionTime']

            except:
                raise ValueError("Could not extract dimensions from file metadata.")

            return dimensions
        
        def validate_dimensions(tif, dimensions):
            image_shape = tif.asarray().shape
            shape_and_dimensions_match = True
            shape_position = 0

            if dimensions['x'] > 1:
                shape_position -= 1
                if not dimensions['x'] == image_shape[shape_position]:
                    shape_and_dimensions_match = False
                    raise ValueError(f"Predicted X-dimension value of {dimensions['x']} but image has X-dimension value of {image_shape[shape_position]} .")


            if dimensions['y'] > 1:
                shape_position -= 1
                if not dimensions['y'] == image_shape[shape_position]:
                    shape_and_dimensions_match = False
                    raise ValueError(f"Predicted Y-dimension value of {dimensions['y']} but image has Y-dimension value of {image_shape[shape_position]} .")

            if dimensions['c'] > 1:
                shape_position -= 1
                if not dimensions['c'] == image_shape[shape_position]:
                    shape_and_dimensions_match = False
                    raise ValueError(f"Predicted Channel-dimension value of {dimensions['c']} but image has Channel-dimension value of {image_shape[shape_position]} .")

            if dimensions['z'] > 1:
                shape_position -= 1
                if not dimensions['z'] == image_shape[shape_position]:
                    shape_and_dimensions_match = False
                    raise ValueError(f"Predicted Z-dimension value of {dimensions['z']} but image has Z-dimension value of {image_shape[shape_position]} .")

            if dimensions['t'] > 1:
                shape_position -= 1
                if not dimensions['t'] == image_shape[shape_position]:
                    shape_and_dimensions_match = False
                    raise ValueError(f"Predicted Time-dimension value of {dimensions['t']} but image has Time-dimension value of {image_shape[shape_position]} .")
         
        

    voxel_size = find_voxel_size(tif)
    dimensions = find_dimensions(tif)

    validate_dimensions(tif, dimensions)

    return voxel_size, dimensions

def format_sig_figs(value, sig_figs=4):
    """Format a number to specified significant figures."""
    if value == 0:
        return 0
    return round(value, -int(np.floor(np.log10(abs(value)))) + (sig_figs - 1))

load_image(file_path)


#A way to input voxels and dimensions manually if cannot find.