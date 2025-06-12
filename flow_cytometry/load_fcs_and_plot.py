print("Running")
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import PolygonSelector, Button
import os
import json
from datetime import datetime
from matplotlib.path import Path
import matplotlib.colors as mcolors

# ========== Configure settings ==========
# Base folder containing all FCS files
base_folder = '/Users/oskar/Desktop/flow_cytometry/SBSE_d5'

# All treatment sample files
treatment_files = {
    'WT': os.path.join(base_folder, 'WT.fcs'),
    'BMH21': os.path.join(base_folder, 'BMH21.fcs'),
    'INK128': os.path.join(base_folder, 'INK128.fcs'),
    'Sal003': os.path.join(base_folder, 'Sal003.fcs')
}

# Reference treatment (for setting thresholds)
reference_treatment = 'WT'

# Output folder
output_folder = base_folder
reference_path = treatment_files[reference_treatment]

# Gates info file path (common across treatments)
gates_info_path = os.path.join(output_folder, 'gates_info.json')
comparison_csv_path = os.path.join(output_folder, 'treatments_comparison.csv')

# Load existing gates info if available
if os.path.exists(gates_info_path):
    with open(gates_info_path, 'r') as f:
        gates_info = json.load(f)
else:
    gates_info = {}

# Load all FCS files
print("Loading treatment files...")
samples = {}
data_by_treatment = {}

for treatment, file_path in treatment_files.items():
    print(f"Loading {treatment} from {file_path}")
    if os.path.exists(file_path):
        try:
            samples[treatment] = FCMeasurement(ID=treatment, datafile=file_path)
            data_by_treatment[treatment] = samples[treatment].data
            
            # Initialize treatment info in gates_info
            if treatment not in gates_info:
                gates_info[treatment] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'gates': {}
                }
        except Exception as e:
            print(f"Error loading {treatment}: {e}")
    else:
        print(f"File not found: {file_path}")

if not data_by_treatment:
    raise ValueError("No valid FCS files could be loaded!")

# Get a reference to the channels from the first loaded sample
reference_data = data_by_treatment[reference_treatment]
print("\nChannels:")
print(reference_data.columns)

# ===== Helper function for interactive gate =====
gate_coords = {}
selected_points = None
vertices_store = None

def interactive_gate_all_treatments(data_dict, x_channel, y_channel, title, log_scale=True):
    """Interactive polygon gate showing all treatments with different colors"""
    global gate_coords, selected_points, vertices_store
    gate_coords.clear()  # reset
    
    # Set up plot with multiple treatments
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Different color for each treatment
    colors = list(mcolors.TABLEAU_COLORS)
    
    # Dictionary to store points and masks for each treatment
    treatment_points = {}
    treatment_masks = {}
    
    # Epsilon for log scaling
    epsilon = 1.0
    
    # Plot each treatment with different color

    for i, (treatment, data) in enumerate(data_dict.items()):
        x = data[x_channel]
        y = data[y_channel]
        
        if log_scale:
            x_plot = np.log10(x + epsilon)
            y_plot = np.log10(y + epsilon)
        else:
            x_plot = x
            y_plot = y
            
        # Apply log scale if requested
        if treatment == 'WT':

            downsample_idx = np.random.choice(len(x), size=len(x)//10, replace=False)
            downsized_x = x[downsample_idx]
            downsized_y = y[downsample_idx]
            
            if log_scale:
                downsized_x_plot = np.log10(downsized_x + epsilon)
                downsized_y_plot = np.log10(downsized_y + epsilon)
            else:
                downsized_x_plot = downsized_x
                downsized_y_plot = downsized_y
       
               # Plot with different color and add to legend
            ax.scatter(downsized_x_plot, downsized_y_plot, s=1, alpha=0.2, 
                   color=colors[i % len(colors)], label=treatment)
            
        # Store the points for later masking
        treatment_points[treatment] = {
            'x': x,
            'y': y,
            'x_plot': x_plot,
            'y_plot': y_plot,
        }
        
    
    if log_scale:
        ax.set_xlabel(f'log10({x_channel})')
        ax.set_ylabel(f'log10({y_channel})')
    else:
        ax.set_xlabel(x_channel)
        ax.set_ylabel(y_channel)
        
    ax.set_title(title + ' (Draw polygon gate, press Done)')
    ax.grid(True)
    ax.legend(loc='upper right')
    
    # Create Done button
    button_ax = fig.add_axes([0.8, 0.01, 0.15, 0.06])
    done_button = Button(button_ax, 'Done', hovercolor='0.975')
    
    # Variable to store polygon vertices
    vertices = []
    
    # PolygonSelector handler
    def onselect(verts):
        nonlocal vertices
        vertices = verts
    
    polygon_selector = PolygonSelector(ax, onselect, useblit=True)
    
    # Button callback
    def done(event):
        nonlocal vertices, treatment_masks
        if len(vertices) < 3:
            print("Please select at least 3 points to form a polygon.")
            return
        
        # For each treatment, apply the same gate
        for treatment, points in treatment_points.items():
            # Create Path object with vertices
            p = Path(vertices)
            
            # Test which points are inside the path
            plot_points = np.column_stack([points['x_plot'], points['y_plot']])
            mask = p.contains_points(plot_points)
            treatment_masks[treatment] = mask
            
            # Calculate stats for this treatment
            if np.any(mask):
                print(f"\n{treatment} gate stats:")
                print(f"Points inside gate: {np.sum(mask)} / {len(mask)} ({100 * np.sum(mask) / len(mask):.2f}%)")
        
        # Store vertices (convert from log scale if needed)
        if log_scale:
            # Convert log-scale vertices to original scale
            original_vertices = [(10**vx - epsilon, 10**vy - epsilon) for vx, vy in vertices]
        else:
            original_vertices = vertices
            
        # Store for access outside the function
        global vertices_store
        vertices_store = original_vertices.copy()  # Make a copy to ensure it's preserved
        
        plt.close(fig)
    
    done_button.on_clicked(done)
    plt.show()
    
    # Create a dict with results for each treatment
    results = {}
    for treatment, mask in treatment_masks.items():
        results[treatment] = {
            'mask': mask, 
            'data': data_dict[treatment][mask],
            'count': int(np.sum(mask)),
            'total': len(mask),
            'percentage': float(100 * np.sum(mask) / len(mask)) if len(mask) > 0 else 0
        }
    
    return results, vertices_store

def compare_fluorescence_ratios(data_dict, reference_treatment, channel, threshold=None, percentile=75):
    """Compare fluorescence/FSC-A ratios across treatments, using reference treatment's threshold"""
    results = {}
    ratio_data = {}
    
    # Calculate ratios for all treatments
    for treatment, data in data_dict.items():
        ratio = data[channel] / data['FSC-A']
        ratio_data[treatment] = ratio
    
    # If no threshold provided, calculate from reference treatment
    if threshold is None:
        threshold = np.percentile(ratio_data[reference_treatment], percentile)
    
    # Create figure with subplots for each treatment
    n_treatments = len(data_dict)
    cols = min(2, n_treatments)
    rows = (n_treatments + cols - 1) // cols  # Ceiling division
    
    fig, axs = plt.subplots(rows, cols, figsize=(12, 5*rows))
    if n_treatments == 1:
        axs = np.array([axs])  # Make it indexable
    axs = axs.flatten()
    
    # Plot each treatment
    for i, (treatment, data) in enumerate(data_dict.items()):
        ax = axs[i]
        ratio = ratio_data[treatment]
        
        # Create mask for values above threshold
        mask = ratio > threshold
        
        # Calculate statistics
        count_above = np.sum(mask)
        total = len(mask)
        percentage = 100 * count_above / total if total > 0 else 0
        
        # Store results
        results[treatment] = {
            'mask': mask,
            'data': data[mask],
            'threshold': float(threshold),
            'count': int(count_above),
            'total': int(total),
            'percentage': float(percentage),
            'ratio_values': ratio.copy()  # Store actual ratio values for potential later use
        }
        
        # Scatter plot colored by whether points are above threshold
        epsilon = 1.0  # For log scale protection
        ax.scatter(np.log10(data['FSC-A'] + epsilon), np.log10(data[channel] + epsilon), 
                  c=mask, s=1, alpha=0.5, cmap='coolwarm')
        
        ax.set_xlabel(f'log10(FSC-A)')
        ax.set_ylabel(f'log10({channel})')
        ax.set_title(f'{treatment}: {percentage:.1f}% above threshold')
        ax.grid(True)
        
        # Print summary
        print(f"\n{treatment} {channel} ratio gate:")
        print(f"Using threshold: {threshold:.6f}")
        print(f"Points above threshold: {count_above} / {total} ({percentage:.2f}%)")
    
    # Hide any unused subplots
    for i in range(n_treatments, len(axs)):
        axs[i].axis('off')
    
    plt.suptitle(f'{channel}/FSC-A Ratio Analysis - Threshold: {threshold:.6f}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    return results, threshold

# ===== Gate 1: Cells (SSC-A vs FSC-A) =====
print("\n===== GATE 1: CELLS (ALL TREATMENTS) =====")
# Create a dictionary with just the needed columns for each treatment
cells_gate_data = {t: df[['FSC-A', 'SSC-A']] for t, df in data_by_treatment.items()}

# Run interactive gating on all treatments
cells_results, cells_vertices = interactive_gate_all_treatments(
    cells_gate_data, 
    'FSC-A', 'SSC-A',
    'Gate 1: Cells (All Treatments)'
)

# Dictionary to store gated cells data for each treatment
cells_by_treatment = {}

# Process and save results for each treatment
if cells_results:
    for treatment, result in cells_results.items():
        # Get the gated data and mask
        cells_data = result['data']
        cells_mask = result['mask']
        
        # Store for next step
        cells_by_treatment[treatment] = data_by_treatment[treatment][cells_mask]
        
        # Save cells data
        cells_output_path = os.path.join(output_folder, f'{treatment}_cells.csv')
        cells_by_treatment[treatment].to_csv(cells_output_path, index=False)
        print(f"{treatment} cells data saved to: {cells_output_path}")
        
        # Store gate info with proper vertices
        gates_info[treatment]['gates']['cells'] = {
            'type': 'polygon',
            'x_channel': 'FSC-A',
            'y_channel': 'SSC-A',
            'vertices': [[float(x), float(y)] for x, y in cells_vertices],  # Convert to list of lists for JSON
            'count': result['count'],
            'total': result['total'],
            'percentage': result['percentage']
        }
    
    # ===== Gate 2: Singlets (FSC-A vs FSC-H) =====
    print("\n===== GATE 2: SINGLETS (ALL TREATMENTS) =====")
    
    # Create dictionary with just the needed columns for singlets gating
    singlets_gate_data = {t: df[['FSC-A', 'FSC-H']] for t, df in cells_by_treatment.items()}
    
    # Run interactive gating on all treatments
    singlets_results, singlets_vertices = interactive_gate_all_treatments(
        singlets_gate_data,
        'FSC-A', 'FSC-H',
        'Gate 2: Singlets (All Treatments)'
    )
    
    # Dictionary to store gated singlets data for each treatment
    singlets_by_treatment = {}
    
    # Process and save results for each treatment
    if singlets_results:
        for treatment, result in singlets_results.items():
            # Get the gated data and mask
            singlets_mask = result['mask']
            
            # Apply mask to the cells data to get singlets
            singlets_data = cells_by_treatment[treatment][singlets_mask]
            singlets_by_treatment[treatment] = singlets_data
            
            # Save singlets data
            singlets_output_path = os.path.join(output_folder, f'{treatment}_singlets.csv')
            singlets_data.to_csv(singlets_output_path, index=False)
            print(f"{treatment} singlets data saved to: {singlets_output_path}")
            
            # Store gate info
            gates_info[treatment]['gates']['singlets'] = {
                'type': 'polygon',
                'x_channel': 'FSC-A',
                'y_channel': 'FSC-H',
                'vertices': [[float(x), float(y)] for x, y in singlets_vertices],  # Convert to list of lists for JSON
                'count': result['count'],
                'total': result['total'],
                'percentage': result['percentage']
            }
        
        # ===== Process fluorescence channels =====
        fluor_channels = ['488nm525-40-A', '561nm610-20-A', '638nm660-10-A']
        fluor_labels = ['488nm525-40 (FITC)', '561nm610-20 (PE-TexRed)', '638nm660-10 (APC)']
        
        print("\n===== FLUORESCENCE CHANNELS ANALYSIS =====")
        
        # Create a comparison dataframe to store all results
        comparison_data = []
        
        # Process each fluorescence channel
        for ch, label in zip(fluor_channels, fluor_labels):
            print(f"\nAnalyzing {label} across all treatments...")
            
            # Compare fluorescence ratios using WT as reference
            fluor_results, threshold = compare_fluorescence_ratios(
                singlets_by_treatment,
                reference_treatment=reference_treatment,
                channel=ch,
                percentile=75  # Top 25%
            )
            
            # Process and save results for each treatment
            for treatment, result in fluor_results.items():
                # Get the gated data and save
                fluor_gated = result['data']
                fluor_output_path = os.path.join(output_folder, f'{treatment}_{ch}_top25pct.csv')
                fluor_gated.to_csv(fluor_output_path, index=False)
                
                # Store gate info
                gates_info[treatment]['gates'][f'{ch}_top25pct'] = {
                    'type': 'ratio',
                    'ratio_channel': ch,
                    'reference_channel': 'FSC-A',
                    'threshold': result['threshold'],
                    'count': result['count'],
                    'total': result['total'],
                    'percentage': result['percentage'],
                    'reference_treatment': reference_treatment
                }
                
                # Add to comparison data
                comparison_data.append({
                    'Treatment': treatment,
                    'Channel': ch,
                    'Display_Name': label,
                    'Threshold': result['threshold'],
                    'Count_Above': result['count'],
                    'Total_Cells': result['total'],
                    'Percentage': result['percentage'],
                    'Reference': treatment == reference_treatment
                })
        
        # Create and save comparison CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_csv_path, index=False)
        print(f"\nComparison data saved to: {comparison_csv_path}")
        
        # Save all gates info to JSON file
        with open(gates_info_path, 'w') as f:
            json.dump(gates_info, f, indent=2)
        print(f"\nGates information saved to: {gates_info_path}")
        
        # Create a summary table of fluorescence channel comparisons
        print("\n===== FLUORESCENCE SUMMARY =====")