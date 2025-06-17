#%% IMPORT PACKAGES

print("Running")
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import PolygonSelector, Button, CheckButtons
import os
import json
from datetime import datetime
from matplotlib.path import Path
import matplotlib.colors as mcolors

#%% LOAD DATA

# ========== Configure settings ==========
# Base folder containing all FCS files
base_folder = os.path.normpath(r'Y:\Room225_SharedFolder\CytoflexLX_data\oskar\relevant_fcs_files\d4_treatments\A')

# All treatment sample files
treatment_files = {
    'untreated': os.path.join(base_folder, 'untreated.fcs'),
    'BMH21': os.path.join(base_folder, 'BMH21.fcs'),
    'INK128': os.path.join(base_folder, 'INK128.fcs'),
    'Sal003': os.path.join(base_folder, 'Sal003.fcs'),
    'MHY1485': os.path.join(base_folder, 'MHY1485.fcs'),
    'puromycin': os.path.join(base_folder, 'puromycin.fcs')
}

# Control files for setting thresholds
control_files = {
    '2iLIF': os.path.join(base_folder, '2iLIF.fcs'),
    'Chiron': os.path.join(base_folder, 'Chiron.fcs'),
    'N2B27': os.path.join(base_folder, 'N2B27.fcs'),
    'untreated_ctrl': os.path.join(base_folder, 'untreated.fcs')  # Using same untreated file as control
}

# Reference treatment (for setting thresholds)
reference_treatment = 'untreated'

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

# Load all FCS files (treatments and controls)
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

print("\nLoading control files...")
data_by_control = {}

for control, file_path in control_files.items():
    print(f"Loading {control} from {file_path}")
    if os.path.exists(file_path):
        try:
            samples[control] = FCMeasurement(ID=control, datafile=file_path)
            data_by_control[control] = samples[control].data
        except Exception as e:
            print(f"Error loading {control}: {e}")
    else:
        print(f"File not found: {file_path}")

if not data_by_treatment:
    raise ValueError("No valid treatment FCS files could be loaded!")

if not data_by_control:
    raise ValueError("No valid control FCS files could be loaded!")

# Get a reference to the channels from the first loaded sample
reference_data = data_by_treatment[reference_treatment]
print("\nChannels:")
print(reference_data.columns)

# After loading FCS files, add:
for treatment, data in data_by_treatment.items():
    print(f"\n{treatment} data shape: {data.shape}")
    print(f"FSC-A range: {data['FSC-A'].min():.2f} to {data['FSC-A'].max():.2f}")
    print(f"SSC-A range: {data['SSC-A'].min():.2f} to {data['SSC-A'].max():.2f}")

for control, data in data_by_control.items():
    print(f"\n{control} (control) data shape: {data.shape}")
    print(f"FSC-A range: {data['FSC-A'].min():.2f} to {data['FSC-A'].max():.2f}")
    print(f"SSC-A range: {data['SSC-A'].min():.2f} to {data['SSC-A'].max():.2f}")

#%% CREATE GATES

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
        
        x_vals = x.values
        y_vals = y.values

        if log_scale:
            x_plot = np.log10(np.maximum(x_vals, epsilon))
            y_plot = np.log10(np.maximum(y_vals, epsilon))
        else:
            x_plot = x_vals
            y_plot = y_vals

        # Downsample if too many points
        if len(x_plot) > 5000:
            downsample_idx = np.random.choice(len(x_plot), size=5000, replace=False)
            x_plot_show = x_plot[downsample_idx]
            y_plot_show = y_plot[downsample_idx]
        else:
            x_plot_show = x_plot
            y_plot_show = y_plot

        # Plot
        ax.scatter(x_plot_show, y_plot_show, s=1, alpha=0.3, 
                color=colors[i % len(colors)], label=treatment)

        # Store the original arrays for later masking
        treatment_points[treatment] = {
            'x': x_vals,
            'y': y_vals,
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

def interactive_fluorescence_gating_with_controls(singlets_data_dict, controls_data_dict, fluor_channels, fluor_labels):
    """Interactive polygon gating on fluorescence channels using controls"""
    
    # Colors for controls
    control_colors = {
        '2iLIF': '#FF6B6B',      # Red
        'Chiron': '#4ECDC4',     # Teal
        'N2B27': '#45B7D1',      # Blue
        'untreated_ctrl': '#96CEB4'  # Green
    }
    
    # Store gates for each channel
    channel_gates = {}
    
    # Process each fluorescence channel separately
    for ch_idx, (channel, label) in enumerate(zip(fluor_channels, fluor_labels)):
        print(f"\n===== GATING {label} =====")
        
        # Create single plot for this channel
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Store plot elements and data for this channel
        plot_elements = {}
        control_visibility = {control: True for control in controls_data_dict.keys()}
        
        # Plot each control
        for control, data in controls_data_dict.items():
            if len(data) == 0:
                continue
                
            # Create scatter plot - handle potential negative values
            epsilon = 1.0
            x_vals = np.log10(np.maximum(data['FSC-A'], epsilon))
            y_vals = np.log10(np.maximum(data[channel], epsilon))
            
            # Downsample if needed
            if len(x_vals) > 3000:
                downsample_idx = np.random.choice(len(x_vals), size=3000, replace=False)
                x_plot = x_vals.iloc[downsample_idx]
                y_plot = y_vals.iloc[downsample_idx]
            else:
                x_plot = x_vals
                y_plot = y_vals
            
            # Create scatter plot
            scatter = ax.scatter(x_plot, y_plot, s=2, alpha=0.6, 
                               color=control_colors[control], label=control)
            
            # Store plot element
            plot_elements[control] = scatter
        
        # Set labels and title
        ax.set_xlabel('log10(FSC-A)')
        ax.set_ylabel(f'log10({channel})')
        ax.set_title(f'{label}\n(Toggle controls, draw polygon gate, press Done)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', markerscale=3)
        
        # Create checkbox widget for toggling controls
        checkbox_ax = fig.add_axes([0.02, 0.7, 0.15, 0.25])
        control_labels = list(controls_data_dict.keys())
        checkbox = CheckButtons(checkbox_ax, control_labels, 
                               [control_visibility[ctrl] for ctrl in control_labels])
        
        # Checkbox callback function
        def toggle_control(label):
            control_visibility[label] = not control_visibility[label]
            if label in plot_elements:
                plot_elements[label].set_visible(control_visibility[label])
            plt.draw()
        
        checkbox.on_clicked(toggle_control)
        
        # Create Done button
        done_ax = fig.add_axes([0.85, 0.02, 0.1, 0.06])
        done_button = Button(done_ax, 'Done', hovercolor='0.975')
        
        # Variables to store gate
        gate_vertices = []
        
        # PolygonSelector for drawing gate
        def onselect(verts):
            nonlocal gate_vertices
            gate_vertices = verts
            print(f"Gate drawn with {len(verts)} vertices")
        
        polygon_selector = PolygonSelector(ax, onselect, useblit=True)
        
        # Done button callback
        gate_finished = False
        def done_callback(event):
            nonlocal gate_finished, gate_vertices
            if len(gate_vertices) < 3:
                print("Please draw a polygon gate with at least 3 points.")
                return
            
            # Convert log-scale vertices back to linear scale
            epsilon = 1.0
            linear_vertices = [(10**vx, 10**vy) for vx, vy in gate_vertices]
            
            # Store the gate
            channel_gates[channel] = {
                'vertices': linear_vertices,
                'log_vertices': gate_vertices,
                'label': label
            }
            
            print(f"Gate saved for {label}")
            gate_finished = True
            plt.close(fig)
        
        done_button.on_clicked(done_callback)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.2, bottom=0.15)
        plt.show()
        
        # Wait for gate to be finished before moving to next channel
        if not gate_finished:
            print(f"No gate was drawn for {label}. Skipping this channel.")
    
    return channel_gates

def apply_gates_to_treatments(singlets_data_dict, channel_gates, fluor_channels, fluor_labels):
    """Apply the polygon gates to treatment data"""
    results = {}
    
    # Create comparison plot
    n_channels = len(channel_gates)
    n_treatments = len(singlets_data_dict)
    
    if n_channels == 0:
        print("No gates were created. Cannot apply to treatments.")
        return results
    
    fig, axes = plt.subplots(n_channels, n_treatments, figsize=(4*n_treatments, 4*n_channels))
    
    # Handle single channel or single treatment cases
    if n_channels == 1 and n_treatments == 1:
        axes = np.array([[axes]])
    elif n_channels == 1:
        axes = axes.reshape(1, -1)
    elif n_treatments == 1:
        axes = axes.reshape(-1, 1)
    
    channel_idx = 0
    for channel, gate_info in channel_gates.items():
        if channel not in fluor_channels:
            continue
            
        results[channel] = {}
        vertices = gate_info['vertices']
        label = gate_info['label']
        
        # Create Path object from vertices (using linear scale vertices)
        gate_path = Path(vertices)
        
        print(f"\n===== Applying {label} gate to treatments =====")
        
        for treat_idx, (treatment, data) in enumerate(singlets_data_dict.items()):
            if n_channels == 1:
                ax = axes[0, treat_idx] if n_treatments > 1 else axes[0]
            else:
                ax = axes[channel_idx, treat_idx] if n_treatments > 1 else axes[channel_idx]
            
            # Test which points are inside the gate (using linear scale coordinates)
            points_to_test = np.column_stack([data['FSC-A'].values, data[channel].values])
            mask = gate_path.contains_points(points_to_test)
            
            # Store results
            results[channel][treatment] = {
                'mask': mask,
                'data': data[mask],
                'gate_vertices': vertices,
                'count': int(np.sum(mask)),
                'total': int(len(mask)),
                'percentage': float(100 * np.sum(mask) / len(mask)) if len(mask) > 0 else 0
            }
            
            # Plot (convert to log scale for visualization)
            epsilon = 1.0
            x_vals = np.log10(np.maximum(data['FSC-A'], epsilon))
            y_vals = np.log10(np.maximum(data[channel], epsilon))
            
            # Color by gate membership
            colors = ['blue' if m else 'lightgray' for m in mask]
            ax.scatter(x_vals, y_vals, c=colors, s=1, alpha=0.5)
            
            # Add gate boundary (convert vertices to log scale for plotting)
            log_vertices = [(np.log10(max(vx, epsilon)), np.log10(max(vy, epsilon))) 
                           for vx, vy in vertices]
            log_vertices.append(log_vertices[0])  # Close the polygon
            gate_x, gate_y = zip(*log_vertices)
            ax.plot(gate_x, gate_y, 'r-', linewidth=2, label='Gate')
            
            # Labels and title
            ax.set_xlabel('log10(FSC-A)')
            ax.set_ylabel(f'log10({channel})')
            percentage = results[channel][treatment]['percentage']
            ax.set_title(f'{treatment}\n{percentage:.1f}% positive')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Print results
            print(f"{treatment}: {results[channel][treatment]['count']} / {results[channel][treatment]['total']} ({percentage:.2f}%) positive")
        
        channel_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    return results

def apply_thresholds_to_treatments(singlets_data_dict, thresholds, fluor_channels):
    """Apply the thresholds set from controls to treatment data"""
    results = {}
    
    # Create comparison plot
    n_channels = len(fluor_channels)
    n_treatments = len(singlets_data_dict)
    
    fig, axes = plt.subplots(n_channels, n_treatments, figsize=(4*n_treatments, 4*n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    if n_treatments == 1:
        axes = axes.reshape(-1, 1)
    
    treatment_colors = list(mcolors.TABLEAU_COLORS)
    
    for ch_idx, channel in enumerate(fluor_channels):
        if channel not in thresholds:
            print(f"Warning: No threshold set for {channel}")
            continue
        
        threshold = thresholds[channel]
        results[channel] = {}
        
        for treat_idx, (treatment, data) in enumerate(singlets_data_dict.items()):
            ax = axes[ch_idx, treat_idx]
            
            # Calculate ratio
            ratio = data[channel] / data['FSC-A']
            mask = ratio > threshold
            
            # Store results
            results[channel][treatment] = {
                'mask': mask,
                'data': data[mask],
                'threshold': float(threshold),
                'count': int(np.sum(mask)),
                'total': int(len(mask)),
                'percentage': float(100 * np.sum(mask) / len(mask)) if len(mask) > 0 else 0
            }
            
            # Plot
            epsilon = 1.0
            x_vals = np.log10(np.maximum(data['FSC-A'], epsilon))
            y_vals = np.log10(np.maximum(data[channel], epsilon))
            
            # Color by threshold
            colors = ['blue' if m else 'lightgray' for m in mask]
            ax.scatter(x_vals, y_vals, c=colors, s=1, alpha=0.5)
            
            # Add threshold line
            x_range = ax.get_xlim()
            x_line = np.linspace(x_range[0], x_range[1], 100)
            y_line = np.log10(threshold) + x_line
            ax.plot(x_line, y_line, 'r--', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('log10(FSC-A)')
            ax.set_ylabel(f'log10({channel})')
            percentage = results[channel][treatment]['percentage']
            ax.set_title(f'{treatment}\n{percentage:.1f}% above threshold')
            ax.grid(True, alpha=0.3)
            
            # Print results
            print(f"\n{treatment} - {channel}:")
            print(f"  Threshold: {threshold:.2e}")
            print(f"  Above threshold: {results[channel][treatment]['count']} / {results[channel][treatment]['total']} ({percentage:.2f}%)")
    
    plt.tight_layout()
    plt.show()
    
    return results

# ===== Gate 1: Cells (SSC-A vs FSC-A) =====
print("\n===== GATE 1: CELLS (ALL TREATMENTS AND CONTROLS) =====")

# Combine treatment and control data for gating
all_data_for_gating = {**data_by_treatment, **data_by_control}
cells_gate_data = {name: df[['FSC-A', 'SSC-A']] for name, df in all_data_for_gating.items()}

# Run interactive gating on all data
cells_results, cells_vertices = interactive_gate_all_treatments(
    cells_gate_data, 
    'FSC-A', 'SSC-A',
    'Gate 1: Cells (All Treatments and Controls)'
)

# Dictionary to store gated cells data
cells_by_treatment = {}
cells_by_control = {}

# Process results
if cells_results:
    for name, result in cells_results.items():
        cells_mask = result['mask']
        
        if name in data_by_treatment:
            cells_by_treatment[name] = data_by_treatment[name][cells_mask]
            
            # Save and store gate info for treatments
            cells_output_path = os.path.join(output_folder, f'{name}_cells.csv')
            cells_by_treatment[name].to_csv(cells_output_path, index=False)
            print(f"{name} cells data saved to: {cells_output_path}")
            
            gates_info[name]['gates']['cells'] = {
                'type': 'polygon',
                'x_channel': 'FSC-A',
                'y_channel': 'SSC-A',
                'vertices': [[float(x), float(y)] for x, y in cells_vertices],
                'count': result['count'],
                'total': result['total'],
                'percentage': result['percentage']
            }
        elif name in data_by_control:
            cells_by_control[name] = data_by_control[name][cells_mask]
    
    # ===== Gate 2: Singlets (FSC-A vs FSC-H) =====
    print("\n===== GATE 2: SINGLETS (ALL TREATMENTS AND CONTROLS) =====")
    
    # Combine cells data for singlets gating
    all_cells_data = {**cells_by_treatment, **cells_by_control}
    singlets_gate_data = {name: df[['FSC-A', 'FSC-H']] for name, df in all_cells_data.items()}
    
    # Run interactive gating
    singlets_results, singlets_vertices = interactive_gate_all_treatments(
        singlets_gate_data,
        'FSC-A', 'FSC-H',
        'Gate 2: Singlets (All Treatments and Controls)'
    )
    
    # Process singlets results
    singlets_by_treatment = {}
    singlets_by_control = {}
    
    if singlets_results:
        for name, result in singlets_results.items():
            singlets_mask = result['mask']
            
            if name in cells_by_treatment:
                singlets_by_treatment[name] = cells_by_treatment[name][singlets_mask]
                
                # Save and store gate info for treatments
                singlets_output_path = os.path.join(output_folder, f'{name}_singlets.csv')
                singlets_by_treatment[name].to_csv(singlets_output_path, index=False)
                print(f"{name} singlets data saved to: {singlets_output_path}")
                
                gates_info[name]['gates']['singlets'] = {
                    'type': 'polygon',
                    'x_channel': 'FSC-A',
                    'y_channel': 'FSC-H',
                    'vertices': [[float(x), float(y)] for x, y in singlets_vertices],
                    'count': result['count'],
                    'total': result['total'],
                    'percentage': result['percentage']
                }
            elif name in cells_by_control:
                singlets_by_control[name] = cells_by_control[name][singlets_mask]
        
        # ===== Interactive Fluorescence Analysis with Controls =====
        print("\n===== FLUORESCENCE ANALYSIS WITH CONTROLS =====")
        
        fluor_channels = ['488nm525-40-A', '561nm610-20-A', '638nm660-10-A']
        fluor_labels = ['488nm525-40 (FITC)', '561nm610-20 (PE-TexRed)', '638nm660-10 (APC)']
        
        # Interactive gating using controls
        channel_gates = interactive_fluorescence_gating_with_controls(
            singlets_by_treatment, 
            singlets_by_control, 
            fluor_channels, 
            fluor_labels
        )
        
        if channel_gates:
            # Apply gates to treatment data
            print("\n===== APPLYING GATES TO TREATMENTS =====")
            fluor_results = apply_gates_to_treatments(
                singlets_by_treatment, 
                channel_gates, 
                fluor_channels, 
                fluor_labels
            )
            
            # Save results and create comparison data
            comparison_data = []
            
            for channel in channel_gates.keys():
                if channel in fluor_results:
                    for treatment, result in fluor_results[channel].items():
                        # Save gated data
                        fluor_gated = result['data']
                        fluor_output_path = os.path.join(output_folder, f'{treatment}_{channel}_positive.csv')
                        fluor_gated.to_csv(fluor_output_path, index=False)
                        
                        # Store gate info
                        gates_info[treatment]['gates'][f'{channel}_positive'] = {
                            'type': 'polygon',
                            'x_channel': 'FSC-A',
                            'y_channel': channel,
                            'vertices': [[float(x), float(y)] for x, y in result['gate_vertices']],
                            'count': result['count'],
                            'total': result['total'],
                            'percentage': result['percentage'],
                            'set_from_controls': True
                        }
                        
                        # Add to comparison data
                        channel_idx = fluor_channels.index(channel) if channel in fluor_channels else 0
                        display_name = fluor_labels[channel_idx] if channel_idx < len(fluor_labels) else channel
                        
                        comparison_data.append({
                            'Treatment': treatment,
                            'Channel': channel,
                            'Display_Name': display_name,
                            'Gate_Type': 'polygon',
                            'Count_Positive': result['count'],
                            'Total_Cells': result['total'],
                            'Percentage': result['percentage'],
                            'Set_From_Controls': True
                        })
            
            # Save comparison data
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_csv(comparison_csv_path, index=False)
                print(f"\nComparison data saved to: {comparison_csv_path}")
            
            # Create consolidated dataset with all cells and their fluorescence status
            print("\n===== CREATING CONSOLIDATED DATASET =====")
            consolidated_data = []
            
            for treatment, data in singlets_by_treatment.items():
                print(f"Processing {treatment} for consolidated dataset...")
                
                # Get fluorescence status for each cell
                fluor_status = {}
                for channel in fluor_channels:
                    if channel in fluor_results and treatment in fluor_results[channel]:
                        fluor_status[channel] = fluor_results[channel][treatment]['mask']
                    else:
                        # If no gate was drawn for this channel, mark all as negative
                        fluor_status[channel] = np.zeros(len(data), dtype=bool)
                
                # Add each cell to consolidated data
                for idx in range(len(data)):
                    cell_data = {
                        'Treatment': treatment,
                        'Cell_ID': f"{treatment}_{idx}",
                        'FSC-A': data['FSC-A'].iloc[idx],
                        'SSC-A': data['SSC-A'].iloc[idx],
                        'FSC-H': data['FSC-H'].iloc[idx],
                        '488nm525-40-A': data['488nm525-40-A'].iloc[idx],
                        '561nm610-20-A': data['561nm610-20-A'].iloc[idx], 
                        '638nm660-10-A': data['638nm660-10-A'].iloc[idx],
                        '488_positive': fluor_status['488nm525-40-A'][idx] if '488nm525-40-A' in fluor_status else False,
                        '561_positive': fluor_status['561nm610-20-A'][idx] if '561nm610-20-A' in fluor_status else False,
                        '638_positive': fluor_status['638nm660-10-A'][idx] if '638nm660-10-A' in fluor_status else False
                    }
                    consolidated_data.append(cell_data)
            
            # Save consolidated dataset
            consolidated_df = pd.DataFrame(consolidated_data)
            consolidated_csv_path = os.path.join(output_folder, 'consolidated_cell_data.csv')
            consolidated_df.to_csv(consolidated_csv_path, index=False)
            print(f"Consolidated cell data saved to: {consolidated_csv_path}")
            print(f"Total cells in consolidated dataset: {len(consolidated_df)}")
            
            # Save gates info
            with open(gates_info_path, 'w') as f:
                json.dump(gates_info, f, indent=2)
            print(f"\nGates information saved to: {gates_info_path}")
            
            print("\n===== ANALYSIS COMPLETE =====")
            print("Polygon gates were drawn using control samples and applied to treatment data.")
            print("Check the saved CSV files for detailed results.")
            print("Positive cells for each channel/treatment have been identified and saved.")
        else:
            print("No gates were created. Analysis incomplete.")
    else:
        print("Singlets gating failed. Cannot proceed with fluorescence analysis.")
else:
    print("Cells gating failed. Cannot proceed with analysis.")