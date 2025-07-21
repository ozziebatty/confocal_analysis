import tifffile
import numpy as np
import napari
import imagej

print("Loading images")
flipped_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_flipped.tiff'
upside_image_pathway = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/20x_example_image_complete.tiff'

upside_image = tifffile.imread(upside_image_pathway)
flipped_image = tifffile.imread(flipped_image_pathway)

# Initialize ImageJ
print("Initialising ImageJ")
ij = imagej.init('sc.fiji:fiji')

# Convert images to Java format
upside = ij.py.to_java(upside_image)
downside = ij.py.to_java(flipped_image)

# Use ImageJ's built-in stitching methods more explicitly
print("Running stitching")
try:
    # Attempt to use a more direct stitching method
    stitched = ij.py.run_macro("""
    run("Pairwise Stitching", "first_image=[upside] second_image=[downside] fusion_method=[Linear Blending] check_peaks=5 compute_overlap");
    """)
    
    # Convert back to Python image
    stitched_img_python = ij.py.from_java(stitched)
    
    # View in Napari
    print("Viewing")
    viewer = napari.Viewer()
    viewer.add_image(stitched_img_python)
    napari.run()

except Exception as e:
    print(f"Stitching failed: {e}")