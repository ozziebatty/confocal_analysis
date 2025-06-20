print("IMPORTING")
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
import tifffile
import numpy as np
import pandas as pd
import threading
from datetime import datetime

# Import the analysis modules at the top
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from cellpose import models

class ImageAnalysisApp:
    def __init__(self):
        # Initialize global state
        self.loaded_project = False
        self.project_path = None
        self.progress_dialog = None
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Image Analysis")
        self.root.geometry("400x350")
        
        # Initialize UI variables
        self.status_var = tk.StringVar(value="No project loaded")
        
        # Create main interface
        self.create_main_interface()
        
    def create_main_interface(self):
        """Create the main application interface"""
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        tk.Label(main_frame, text="Image Analysis Tool", font=("Arial", 14, "bold")).pack(pady=(0, 15))

        # Project buttons
        project_frame = tk.Frame(main_frame)
        project_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(project_frame, text="Create Project", command=self.create_project_window, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(project_frame, text="Load Project", command=self.load_project, width=15).pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = tk.LabelFrame(main_frame, text="Project Status")
        status_frame.pack(fill=tk.X, pady=10, padx=5)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var, justify=tk.LEFT, padx=10, pady=5)
        status_label.pack(fill=tk.X)
        
        # Action buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.choose_processes_btn = tk.Button(button_frame, text="Select segmentation options", 
                                            command=self.choose_processes_window, width=42, state=tk.DISABLED)
        self.choose_processes_btn.pack(pady=3)
        
        self.define_parameters_btn = tk.Button(button_frame, text="Define Parameters", 
                                             command=self.define_parameters_window, width=42, state=tk.DISABLED)
        self.define_parameters_btn.pack(pady=3)
        
        # Fixed: Pass function reference, not function call
        self.run_analysis_btn = tk.Button(button_frame, text="Run Analysis", 
                                        command=self.run_analysis_window, width=42, state=tk.DISABLED)
        self.run_analysis_btn.pack(pady=3)

    def update_main_page(self):
        """Update the main page based on project status"""
        if self.loaded_project:
            project_name = os.path.basename(self.project_path)
            self.status_var.set(f"Project: {project_name}\nLocation: {self.project_path}")
            
            # Enable buttons
            for btn in [self.choose_processes_btn, self.define_parameters_btn, self.run_analysis_btn]:
                btn.config(state=tk.NORMAL)
        else:
            self.status_var.set("No project loaded")
            
            # Disable buttons
            for btn in [self.choose_processes_btn, self.define_parameters_btn, self.run_analysis_btn]:
                btn.config(state=tk.DISABLED)

    def load_project(self):
        """Load an existing project"""
        project_path = filedialog.askdirectory(title="Select Project Folder")
        if not project_path:
            return
            
        project_path = os.path.normpath(project_path)
        
        # Simple validation - check if the folder exists
        if not os.path.isdir(project_path):
            messagebox.showerror("Error", "Invalid folder.")
            return
            
        self.loaded_project = True
        self.project_path = project_path
        self.update_main_page()

    def create_project_window(self):
        """Create a new project"""
        project_window = tk.Toplevel(self.root)
        project_window.title("Create Project")
        project_window.geometry("700x550")
        
        frame = tk.Frame(project_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Input file/folder
        input_frame = tk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=5)
        tk.Label(input_frame, text="Input File/Folder:").pack(side=tk.LEFT)
        file_entry = tk.Entry(input_frame, width=40)
        file_entry.pack(side=tk.LEFT, padx=5)
        
        def browse_input():
            path = filedialog.askdirectory()
            if path:
                file_entry.delete(0, tk.END)
                file_entry.insert(0, os.path.normpath(path))

        tk.Button(input_frame, text="Browse", command=browse_input).pack(side=tk.LEFT)
        
        # Destination folder
        dest_frame = tk.Frame(frame)
        dest_frame.pack(fill=tk.X, pady=5)
        tk.Label(dest_frame, text="Destination Folder:").pack(side=tk.LEFT)
        dest_folder_entry = tk.Entry(dest_frame, width=40)
        dest_folder_entry.pack(side=tk.LEFT, padx=5)
        
        def browse_destination():
            path = filedialog.askdirectory()
            if path:
                dest_folder_entry.delete(0, tk.END)
                dest_folder_entry.insert(0, os.path.normpath(path))

        tk.Button(dest_frame, text="Browse", command=browse_destination).pack(side=tk.LEFT)
        
        # Segmentation channel option
        segmentation_frame = tk.Frame(frame)
        segmentation_frame.pack(fill=tk.X, pady=5)
        
        contains_segmentation_var = tk.BooleanVar()
        tk.Checkbutton(segmentation_frame, text="Contains Segmentation Channel", 
                      variable=contains_segmentation_var).pack(anchor="w")
        
        segmentation_var = tk.IntVar(value=-1)
        
        # Channel entries
        channels_frame = tk.LabelFrame(frame, text="Channel Names")
        channels_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        channel_entries = []
        radio_buttons = []
        
        for i in range(8):
            channel_frame = tk.Frame(channels_frame)
            channel_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(channel_frame, text=f"Channel {i+1}:", width=10).pack(side=tk.LEFT)
            entry = tk.Entry(channel_frame, width=25)
            entry.pack(side=tk.LEFT, padx=5)
            channel_entries.append(entry)
            
            radio = tk.Radiobutton(channel_frame, text="Segmentation", 
                                 variable=segmentation_var, value=i)
            radio.pack(side=tk.LEFT)
            radio_buttons.append(radio)
            radio.config(state=tk.DISABLED)
        
        # Function to toggle radio buttons
        def toggle_segmentation_options():
            state = tk.NORMAL if contains_segmentation_var.get() else tk.DISABLED
            for rb in radio_buttons:
                rb.config(state=state)
        
        contains_segmentation_var.trace("w", lambda *args: toggle_segmentation_options())
        
        def create_project():
            input_path = file_entry.get()
            dest_folder = dest_folder_entry.get()
            
            if not input_path or not dest_folder:
                messagebox.showerror("Input Error", "Please provide both input file/folder and destination folder.")
                return

            channel_names = [entry.get().strip() for entry in channel_entries]
            if all(not name for name in channel_names):
                messagebox.showerror("Input Error", "At least one channel name field must be filled.")
                return

            if contains_segmentation_var.get():
                segmentation_index = segmentation_var.get()
                if segmentation_index == -1 or not channel_names[segmentation_index]:
                    messagebox.showerror("Input Error", "Segmentation channel must be a valid, non-empty channel.")
                    return

            # Create destination if it doesn't exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Copy input file/folder
            try:
                if os.path.isdir(input_path):
                    shutil.copytree(input_path, os.path.join(dest_folder, os.path.basename(input_path)))
                else:
                    shutil.copy(input_path, dest_folder)
            except Exception as e:
                messagebox.showerror("Copy Error", f"Error copying files: {str(e)}")
                return

            # Save channel details
            try:
                with open(os.path.join(dest_folder, "channel_details.csv"), mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["channel", "segmentation_channel"])
                    for idx, name in enumerate(channel_names):
                        if name:  # Only write non-empty channel names
                            writer.writerow([name, 'yes' if segmentation_var.get() == idx else 'No'])
            except Exception as e:
                messagebox.showerror("File Error", f"Error saving channel details: {str(e)}")
                return

            # Update app state
            self.loaded_project = True
            self.project_path = dest_folder
            self.update_main_page()

            messagebox.showinfo("Success", f"Project created successfully in {dest_folder}")
            project_window.destroy()
        
        # Buttons
        button_frame = tk.Frame(frame)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Cancel", command=project_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Create Project", command=create_project, width=15).pack(side=tk.LEFT, padx=5)

    def choose_processes_window(self):
        """Choose segmentation processes"""
        if not self.loaded_project:
            messagebox.showerror("Error", "Please load a project first.")
            return
            
        processes_csv_path = os.path.join(self.project_path, 'processes.csv')
       
        process_window = tk.Toplevel(self.root)
        process_window.title("Choose Segmentation")
        process_window.geometry("400x300")
       
        frame = tk.Frame(process_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
       
        tk.Label(frame, text="Select Segmentation Option", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Segmentation frame with radio buttons
        segmentation_frame = tk.LabelFrame(frame, text="Segmentation")
        segmentation_frame.pack(fill=tk.X, pady=10)
       
        segmentation_choice = tk.StringVar(value="none")
       
        tk.Radiobutton(segmentation_frame, text="No segmentation",
                      variable=segmentation_choice, value="none").pack(anchor='w', pady=2)
        tk.Radiobutton(segmentation_frame, text="Cellpose nuclear segmentation",
                      variable=segmentation_choice, value="nuclear").pack(anchor='w', pady=2)
        tk.Radiobutton(segmentation_frame, text="Cellpose cellular segmentation",
                      variable=segmentation_choice, value="cellular").pack(anchor='w', pady=2)
       
        def save_processes():
            seg_choice = segmentation_choice.get()
            nuclear_selected = (seg_choice == "nuclear")
            cellular_selected = (seg_choice == "cellular")
           
            # Write to CSV
            with open(processes_csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["category", "process", "selected"])
                writer.writerow(["segmentation", "cellpose_nuclear_segmentation", "yes" if nuclear_selected else "no"])
                writer.writerow(["segmentation", "cellpose_cellular_segmentation", "yes" if cellular_selected else "no"])
           
            if seg_choice == "none":
                messagebox.showwarning("Warning", "No segmentation process selected.")
                return
           
            messagebox.showinfo("Process Saved", "Selected segmentation process has been saved to CSV.")
            process_window.destroy()
       
        button_frame = tk.Frame(frame)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Cancel", command=process_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save", command=save_processes, width=10).pack(side=tk.LEFT, padx=5)

    def define_parameters_window(self):
        """Define processing parameters using Napari interface"""
        if not self.project_path:
            messagebox.showerror("Error", "Please load a project first.")
            return
        
        # This would contain your existing define_parameters_window code
        # For now, just show a placeholder
        messagebox.showinfo("Parameters", "Parameters definition window would open here.\nThis requires Napari and the full parameter definition code.")

    def run_analysis_window(self):
        """Run the complete analysis pipeline"""
        if not self.project_path:
            messagebox.showerror("Error", "Please load a project first.")
            return

        # Create analysis options window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Run Analysis")
        analysis_window.geometry("400x300")

        frame = tk.Frame(analysis_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame, text="Run Analysis", font=("Arial", 12, "bold")).pack(pady=10)

        processes_frame = tk.LabelFrame(frame, text="Select Processes")
        processes_frame.pack(fill=tk.X, pady=10)

        processes = {
            "Preprocessing": tk.BooleanVar(value=True),
            "Segmentation": tk.BooleanVar(value=True),
            "Analysis": tk.BooleanVar(value=True)
        }

        for process, var in processes.items():
            tk.Checkbutton(processes_frame, text=process, variable=var).pack(anchor="w", padx=10)

        def start_analysis():
            analysis_window.destroy()
            try:
                # Load project data and start processing
                result = self.load_project_data(processes)
                if result is None:
                    return
                    
                selected_processes, channel_parameters, global_parameters, segmentation_channel_index, tiff_files, total_files = result

                # Create progress dialog
                progress_dialog = ProgressDialog(
                    parent=self.root,
                    title="Processing Images", 
                    total_images=total_files,
                )

                # Start processing in separate thread
                processing_thread = threading.Thread(
                    target=self.process_all_images,
                    args=(selected_processes, channel_parameters, global_parameters, 
                          segmentation_channel_index, tiff_files, total_files, progress_dialog)
                )
                processing_thread.daemon = True
                processing_thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")

        button_frame = tk.Frame(frame)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Cancel", command=analysis_window.destroy, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Start", command=start_analysis, width=10).pack(side=tk.LEFT, padx=5)

    def load_project_data(self, processes):
        """Load and validate project data"""
        # Check if segmentation is selected from processes.csv
        processes_path = os.path.join(self.project_path, 'processes.csv')
        if os.path.exists(processes_path):
            processes_df = pd.read_csv(processes_path)
            segmentation_row = processes_df[processes_df['process'] == 'cellpose_nuclear_segmentation']
            segmentation_selected = len(segmentation_row) > 0 and segmentation_row['selected'].values[0] == 'yes'
        else:
            segmentation_selected = True

        # Load parameters from parameters.csv
        parameters_path = os.path.join(self.project_path, 'parameters.csv')
        if os.path.exists(parameters_path):
            parameters_df = pd.read_csv(parameters_path)
            
            try:
                required_columns = ['parameter', 'process', 'channel', 'value', 'default_value', 'data_type', 'must_be_odd']
                if not all(col in parameters_df.columns for col in required_columns):
                    raise ValueError(f"Parameters CSV must contain columns: {required_columns}")
            except Exception as e:
                raise ValueError(f"Error loading parameters CSV file: {str(e)}")
            
            channel_parameters = {}
            global_parameters = {}
            
            # Initialize parameter values from the CSV
            for _, row in parameters_df.iterrows():
                parameter_name = row['parameter']
                channel = row['channel']
                value = row['value']
                default_value = row['default_value']
                
                parameter_value = value if pd.notnull(value) else default_value
                
                if pd.notnull(channel):
                    try:
                        channel = int(channel)
                        if channel not in channel_parameters:
                            channel_parameters[channel] = {}
                        channel_parameters[channel][parameter_name] = parameter_value
                    except ValueError:
                        global_parameters[parameter_name] = parameter_value
                else:
                    global_parameters[parameter_name] = parameter_value

        # Find segmentation channel
        segmentation_channel_index = self.find_segmentation_channel()

        # Check for TIFF files
        tiff_files = [f for f in os.listdir(self.project_path) 
                     if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]

        if not tiff_files:
            messagebox.showerror("Error", "No TIFF files found in the project folder.")
            return None

        total_files_to_analyse = len(tiff_files)

        # Get selected processes
        selected_processes = [p for p, v in processes.items() if v.get()]
        if not selected_processes:
            messagebox.showerror("Error", "Please select at least one process.")
            return None

        return selected_processes, channel_parameters, global_parameters, segmentation_channel_index, tiff_files, total_files_to_analyse

    def find_segmentation_channel(self):
        """Find the segmentation channel from channel_details.csv"""
        channel_details_path = os.path.join(self.project_path, 'channel_details.csv')
        try:
            df = pd.read_csv(channel_details_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find file: {channel_details_path}")
        
        if 'channel' not in df.columns or 'segmentation_channel' not in df.columns:
            raise ValueError("CSV must have 'channel' and 'segmentation_channel' columns")
        
        segmentation_rows = df[df['segmentation_channel'] == 'yes']
        if len(segmentation_rows) != 1:
            raise ValueError("Must have exactly one segmentation channel marked as 'yes'")
        
        return segmentation_rows.index[0]

    def process_all_images(self, selected_processes, channel_parameters, global_parameters, 
                          segmentation_channel_index, tiff_files, total_files, progress_dialog):
        """Process all images in the project"""
        for file_index, tiff_file in enumerate(tiff_files):
            # Check for cancellation
            if progress_dialog.is_cancelled():
                break
                
            # Update progress
            progress_dialog.update_image_progress(file_index + 1, total_files)

            # Set up paths
            tiff_path = os.path.join(self.project_path, tiff_file)
            output_folder = os.path.join(self.project_path, os.path.splitext(tiff_file)[0])
            
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)

            # Load and process image
            try:
                image = tifffile.imread(tiff_path)
                self.process_single_image(
                    image, segmentation_channel_index, selected_processes, 
                    channel_parameters, global_parameters, output_folder, 
                    tiff_file, progress_dialog
                )
            except Exception as e:
                print(f"Error processing {tiff_file}: {str(e)}")
                continue
        
        # Close progress dialog when done
        if progress_dialog:
            progress_dialog.close()

    def process_single_image(self, image, segmentation_channel_index, selected_processes, 
                           channel_parameters, global_parameters, output_folder, 
                           tiff_file, progress_dialog):
        preprocessed_image = image.copy()
        segmented_image = None
        file_base_name = os.path.splitext(tiff_file)[0]

        preprocessed_path = os.path.join(output_folder, f"{file_base_name}_preprocessed.tiff")
        raw_segmentation_path = os.path.join(output_folder, f"{file_base_name}_raw_segmentation.tiff")
        segmentation_path = os.path.join(output_folder, f"{file_base_name}_segmentation.tiff")
        characterised_cells_path = os.path.join(output_folder, f"{file_base_name}_characterised_cells.csv")


        # PREPROCESSING
        if "Preprocessing" in selected_processes and not progress_dialog.is_cancelled():
            print(f"{datetime.now():%H:%M:%S} - Starting preprocessing for {tiff_file}")
            
            total_channels = image.shape[1] if len(image.shape) > 3 else 1

            for channel in range(total_channels):
                if progress_dialog.is_cancelled():
                    break
                    
                # Update progress
                progress_dialog.update_process_progress("preprocessing", 
                                                      ((channel + 1) * 100) // total_channels)
                
                # Get channel-specific parameters
                this_channel_parameters = channel_parameters.get(channel, {})
                
                # Process channel slice
                channel_slice = image[:, channel, :, :]
                preprocessed_image[:, channel, :, :] = self.preprocess_channel(
                    channel_slice, this_channel_parameters, progress_dialog
                )
            
            # Save preprocessed image if not cancelled
            if not progress_dialog.is_cancelled():
                try:
                    tifffile.imwrite(preprocessed_path, preprocessed_image)
                    print(f"Saved preprocessed image: {preprocessed_path}")
                except Exception as e:
                    print(f"Error saving preprocessed image: {str(e)}")
        else:
            # Check if preprocessed file already exists and load it
            if os.path.exists(preprocessed_path):
                try:
                    preprocessed_image = tifffile.imread(preprocessed_path)
                    print(f"Loaded existing preprocessed image: {preprocessed_path}")
                except Exception as e:
                    print(f"Error loading preprocessed image: {str(e)}")
                    preprocessed_image = image.copy()

        # SEGMENTATION
        if "Segmentation" in selected_processes and not progress_dialog.is_cancelled():
            print(f"{datetime.now():%H:%M:%S} - Starting segmentation for {tiff_file}")
            
            # Use preprocessed image if available, otherwise use original
            if preprocessed_image is not None:
                segmentation_input = preprocessed_image[:, segmentation_channel_index, :, :]
            else:
                segmentation_input = image[:, segmentation_channel_index, :, :]

            segmentation_parameters = global_parameters

            # Run segmentation
            try:
                raw_segmented_image, segmented_image = self.segment_and_stitch(
                    segmentation_input, segmentation_parameters, progress_dialog
                )
                
                # Save segmentation results if not cancelled
                if not progress_dialog.is_cancelled() and segmented_image is not None:
                    try:
                        tifffile.imwrite(segmentation_path, segmented_image.astype(np.uint16))
                        tifffile.imwrite(raw_segmentation_path, raw_segmented_image.astype(np.uint16))
                        print(f"Saved segmentation results: {segmentation_path}")
                    except Exception as e:
                        print(f"Error saving segmentation results: {str(e)}")
            except Exception as e:
                print(f"Error running segmentation: {str(e)}")
        else:
            # Check if preprocessed file already exists and load it
            if os.path.exists(segmentation_path):
                try:
                    segmented_image = tifffile.imread(segmentation_path)
                    print(f"Loaded existing segmented image: {segmentation_path}")
                except Exception as e:
                    print(f"Error loading segmented image: {str(e)}")             

        # ANALYSIS
        if "Analysis" in selected_processes and not progress_dialog.is_cancelled():
            print(f"{datetime.now():%H:%M:%S} - Starting analysis for {tiff_file}")
            
            try:
                # Check if we have segmentation results for analysis
                if segmented_image is not None:
                    characterised_cells = self.characterise_cells(
                        preprocessed_image if preprocessed_image is not None else image,
                        segmented_image,
                        progress_dialog
                    )
                    
                    # Save analysis results
                    if not progress_dialog.is_cancelled():
                        try:
                            characterised_cells.to_csv(characterised_cells_path, index=False)
                            print(f"Saved analysis results: {characterised_cells_path}")
                        except Exception as e:
                            print(f"Error saving analysis results: {str(e)}")
                else:
                    print("No segmentation available for analysis")
                    
            except Exception as e:
                print(f"Error during analysis: {str(e)}")

        print(f"{datetime.now():%H:%M:%S} - Completed processing for {tiff_file}")
        return preprocessed_image, segmented_image

    def preprocess_channel(self, channel_slice, channel_parameters, progress_dialog):
        """Apply preprocessing to a single channel"""
        image = channel_slice.copy()

        # Get parameters with defaults
        clahe_kernel_size = int(channel_parameters.get('CLAHE_kernel_size', 16))
        clahe_clip_limit = float(channel_parameters.get('CLAHE_clip_limit', 0.005))
        clahe_n_bins = int(channel_parameters.get('CLAHE_n_bins', 11))
        
        gaussian_sigma = float(channel_parameters.get('gaussian_sigma', 0.4))
        gaussian_kernel_size = float(channel_parameters.get('gaussian_kernel_size', 11))

        def apply_gaussian(image, sigma, kernel_size):
            
            # Ensure kernel size is odd
            kernel_size = int(kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            gaussian_blurred_image = img_as_ubyte(gaussian(
                img_as_float(image),
                sigma=sigma,
                truncate=kernel_size))
            return gaussian_blurred_image

        def apply_CLAHE(image, kernel_size, clip_limit, n_bins):
                
            # Ensure kernel size is valid
            kernel_size = max(3, int(kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            CLAHE_image = img_as_ubyte(equalize_adapthist(
                img_as_float(image),
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=n_bins
            ))

            return CLAHE_image

        # Step 1: Gaussian
        if not progress_dialog.is_cancelled():
            progress_dialog.update_process_progress("preprocessing", 25)
            image = apply_gaussian(image, gaussian_sigma, gaussian_kernel_size)

        # Step 2: CLAHE
        if not progress_dialog.is_cancelled():
            progress_dialog.update_process_progress("preprocessing", 75)
            image = apply_CLAHE(image, clahe_kernel_size, clahe_clip_limit, clahe_n_bins)

        return image

    def segment_and_stitch(self, channel_slice, segmentation_parameters, progress_dialog):
        """Perform segmentation and stitching on a channel"""
        # Get parameters with defaults
        cell_diameter = float(segmentation_parameters.get('cell_diameter', 8.0))
        flow_threshold = float(segmentation_parameters.get('flow_threshold', 0.5))
        cellprob_threshold = float(segmentation_parameters.get('cellprob_threshold', 0.5))
        iou_threshold = 0.5

        def segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold):
            """Segment nuclei in 2D slices using Cellpose"""
            try:
                model = models.Cellpose(gpu=False, model_type='nuclei')
                
                total_z = channel_slice.shape[0]
                segmented_image = np.zeros_like(channel_slice, dtype=np.uint16)
                total_cells_segmented = 0
                
                for z in range(total_z):
                    if progress_dialog.is_cancelled():
                        break
                        
                    # Update progress
                    progress_dialog.update_process_progress("segmentation", 
                                                          (z * 50) // total_z)
                    
                    z_slice = channel_slice[z]
                    
                    # Ensure proper data type for Cellpose
                    if z_slice.dtype != np.uint8:
                        z_slice = img_as_ubyte(img_as_float(z_slice))
                    
                    segmented_image_z_slice, flows, styles, diams = model.eval(
                        z_slice,
                        diameter=cell_diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                    )
                
                    segmented_image[z] = segmented_image_z_slice
                    total_cells_segmented += len(np.unique(segmented_image_z_slice)) - 1
                
                print(f"Total cells segmented: {total_cells_segmented}")
                return segmented_image
                
            except Exception as e:
                print(f"Error during 2D segmentation: {str(e)}")
                return np.zeros_like(channel_slice, dtype=np.uint16)

        def stitch_by_iou(segmented_image):
            """Stitch 2D segmentation masks using IoU"""
            try:
                def calculate_iou(cell_1, cell_2):
                    """Calculate Intersection over Union (IoU) between two binary masks"""
                    intersection = np.logical_and(cell_1, cell_2).sum()
                    union = np.logical_or(cell_1, cell_2).sum()
                    if union == 0:
                        return 0
                    return intersection / union
                
                total_z = segmented_image.shape[0]
                stitched_image = segmented_image.copy().astype(np.uint16)
                current_label = 1
                
                for z in range(1, total_z):
                    if progress_dialog.is_cancelled():
                        break
                        
                    # Update progress (50-90% of segmentation)
                    progress_dialog.update_process_progress("stitching", 
                                                          50 + ((z-1) * 40) // (total_z-1))

                    previous_slice = stitched_image[z-1]
                    current_slice = stitched_image[z]
                    
                    new_labels = np.zeros_like(current_slice)
                    unique_labels = np.unique(current_slice)
                    
                    for label in unique_labels:
                        if label == 0:
                            continue

                        current_cell = current_slice == label
                        
                        max_iou = 0
                        best_match_label = 0
                        overlap_labels = np.unique(previous_slice[current_cell])
                        overlap_labels = overlap_labels[overlap_labels > 0]
                        
                        for previous_label in overlap_labels:
                            previous_cell = previous_slice == previous_label
                            iou = calculate_iou(current_cell, previous_cell)
                            if iou > max_iou:
                                max_iou = iou
                                best_match_label = previous_label
                        
                        if max_iou >= iou_threshold:
                            new_labels[current_cell] = best_match_label
                        else:
                            new_labels[current_cell] = current_label
                            current_label += 1
                    
                    stitched_image[z] = new_labels

                return stitched_image
                
            except Exception as e:
                print(f"Error during stitching: {str(e)}")
                return segmented_image

        def clean_labels(stitched_image):
            """Relabel segmented image so that every cell has a unique label"""
            try:
                unique_labels = np.unique(stitched_image)
                unique_labels = unique_labels[unique_labels != 0]
                
                label_mapping = {0: 0}
                for new_label, old_label in enumerate(unique_labels, start=1):
                    label_mapping[old_label] = new_label
                
                relabeled_image = np.zeros_like(stitched_image)
                for old_label, new_label in label_mapping.items():
                    relabeled_image[stitched_image == old_label] = new_label
                
                return relabeled_image
                
            except Exception as e:
                print(f"Error during label cleaning: {str(e)}")
                return stitched_image

        # Perform segmentation pipeline
        if progress_dialog.is_cancelled():
            return None, None
            
        segmented_image = segment_2D(channel_slice, cell_diameter, flow_threshold, cellprob_threshold)
        
        if progress_dialog.is_cancelled():
            return segmented_image, None
            
        stitched_image = stitch_by_iou(segmented_image)
        
        if progress_dialog.is_cancelled():
            return segmented_image, stitched_image
            
        # Final progress update
        progress_dialog.update_process_progress("stitching", 90)
        cleaned_segmented_image = clean_labels(stitched_image)
        progress_dialog.update_process_progress("stitching", 100)

        return segmented_image, cleaned_segmented_image

    def characterise_cells(self, image, segmented_image, progress_dialog):
        channel_names = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']

        total_z = image.shape[0]
        total_channels = image.shape[1]
        y_pixels = image.shape[2]
        x_pixels = image.shape[3]

        total_cells = np.max(segmented_image) + 1 #Include a background for indexing (so label 1 at position 1)

        characterised_cells = {cell: {'pixel_count': 0, 'z_position': 0.0, **{channel: 0.0 for channel in channel_names}} for cell in range(total_cells)}

        def quantify_cell_fluorescence(image, segmented_image, characterised_cells):
            #Initialise arrays
            channel_values = np.zeros(total_cells, dtype=[('cell_number', int), ('pixel_count', int)] + [(channel, float) for channel in channel_names])
            z_slice_fluorescence = np.zeros(total_z, dtype=[('channels', float, total_channels), ('pixel_count', int)])
            
            for z in range(total_z):
                z_slice_image = image[z]
                z_slice_segmented_image = segmented_image[z]
                total_cells_in_z_slice = len(np.unique(z_slice_segmented_image))

                for label in np.unique(z_slice_segmented_image):
                    if label == 0:
                        continue

                    channel_values[label]['cell_number'] = label           

                    #Mask cell
                    masked_cell = (z_slice_segmented_image == label)
                    
                    # Calculate sum of channel intensities for the cell
                    channel_sums = [np.sum(z_slice_image[channel][masked_cell]) for channel in range(total_channels)]
                    
                    # Calculate pixel count for the cell
                    running_pixel_count = np.sum(masked_cell) 
                    
                    # If this cell has already been partially processed in a previous slice, accumulate data
                    for channel, channel_sum in zip(channel_names, channel_sums):
                        channel_values[label][channel] += channel_sum
                    channel_values[label]['pixel_count'] += running_pixel_count
                    z_slice_fluorescence[z]['channels'] += np.array(channel_sums)
                    z_slice_fluorescence[z]['pixel_count'] += running_pixel_count

            #Average channel intensities by pixel count
            for cell_data in channel_values:
                cell_label = cell_data['cell_number']
                pixel_count = cell_data['pixel_count']
                if pixel_count > 0:
                    characterised_cells[cell_label]['pixel_count'] = pixel_count
                    for channel in channel_names:
                        average_channel_value = cell_data[channel] / pixel_count
                        characterised_cells[cell_label][channel] = average_channel_value

            for z_slice in z_slice_fluorescence:
                z_slice['channels'] /= z_slice['pixel_count']

            return z_slice_fluorescence

        def find_average_z_slice_of_each_label(segmented_image, characterised_cells):
            z_slice_averages = np.zeros(total_cells, dtype=[('running_z_total', int), ('z_stack_count', int), ('average_z', float)])

            for z in range(total_z):
                z_slice_segmented_image = segmented_image[z]
                total_cells_in_z_slice = len(np.unique(z_slice_segmented_image))

                for label in np.unique(z_slice_segmented_image):
                    z_slice_averages[label]['running_z_total'] += z
                    z_slice_averages[label]['z_stack_count'] += 1

            for label in range(1, total_cells):
                z_slice_averages[label]['average_z'] = z_slice_averages[label]['running_z_total'] / z_slice_averages[label]['z_stack_count']

            for label in range(len(z_slice_averages)):
                characterised_cells[label]['z_position'] = z_slice_averages[label]['average_z']

            # Save the DataFrame to a CSV file

        z_slice_averages = find_average_z_slice_of_each_label(segmented_image, characterised_cells)

        z_slice_fluorescence = quantify_cell_fluorescence(image, segmented_image, characterised_cells)

        characterised_cells_df = pd.DataFrame.from_dict(characterised_cells, orient='index')
        characterised_cells_df.reset_index(inplace=True)
        characterised_cells_df.rename(columns={'index': 'cell_number'}, inplace=True)

        return characterised_cells_df


    def run(self):
        """Start the application"""
        self.root.mainloop()


class ProgressDialog:
    """Progress dialog for image processing"""
    def __init__(self, parent, title, total_images):
        self.parent = parent
        self.total_images = total_images
        self.current_image = 0
        self.cancelled = False
        
        # Create dialog window
        self.root = tk.Toplevel(parent)
        self.root.title(title)
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        self.root.transient(parent)
        self.root.grab_set()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create progress dialog widgets"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Overall image progress
        ttk.Label(main_frame, text="Overall Progress").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.image_progress_text = ttk.Label(main_frame, text=f"Image 0/{self.total_images}")
        self.image_progress_text.grid(row=0, column=1, sticky=tk.E, pady=(0, 5))
        
        self.image_progress = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=self.total_images)
        self.image_progress.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 15))
        
        # Process progress bars
        processes = ["Preprocessing", "Segmentation", "Stitching", "Analysis"]
        self.process_progress = {}
        
        for i, process in enumerate(processes):
            row_offset = 2 + i * 2
            ttk.Label(main_frame, text=process).grid(row=row_offset, column=0, sticky=tk.W, pady=(0, 5))
            progress_bar = ttk.Progressbar(main_frame, length=460, mode="determinate", maximum=100)
            progress_bar.grid(row=row_offset + 1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
            self.process_progress[process.lower()] = progress_bar
        
        # Cancel button
        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.confirm_cancel)
        self.cancel_button.grid(row=10, column=0, columnspan=2, pady=(0, 10))

    def update_image_progress(self, step, total_steps):
        """Update image progress"""
        self.image_progress["value"] = step
        self.image_progress_text.config(text=f"Image {step}/{total_steps}")
        self.reset_process_progress()
        self.root.update_idletasks()

    def reset_process_progress(self):
        """Reset all process progress bars"""
        for progress_bar in self.process_progress.values():
            progress_bar["value"] = 0
        
    def update_process_progress(self, process_name, percent):
        """Update specific process progress"""
        percent = min(100, max(0, percent))
        process_key = process_name.lower()
        
        if process_key in self.process_progress:
            self.process_progress[process_key]["value"] = percent
        
        self.root.update_idletasks()

    def confirm_cancel(self):
        """Show confirmation dialog before cancelling"""
        if messagebox.askyesno("Cancel Processing", 
                            "Are you sure you want to cancel?\nOnly fully processed images will be saved."):
            self.cancelled = True
            self.cancel_button.config(state=tk.DISABLED)
            self.root.update_idletasks()
            self.close()
            
    def is_cancelled(self):
        """Check if processing was cancelled"""
        return self.cancelled
    
    def on_closing(self):
        """Handle window close button"""
        self.confirm_cancel()
        
    def close(self):
        """Close the dialog"""
        self.root.grab_release()
        self.root.destroy()


# Run the application
if __name__ == "__main__":
    app = ImageAnalysisApp()
    app.run()