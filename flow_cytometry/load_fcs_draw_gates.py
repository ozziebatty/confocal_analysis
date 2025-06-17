#%% IMPORT PACKAGES

print("Running Gate Definition Script")
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

#%% CONFIGURE SETTINGS

# ========== Configure settings ==========
# Base folder containing reference FCS files for gate definition
reference_folder = os.path.normpath(r'Y:\Room225_SharedFolder\CytoflexLX_data\oskar\relevant_fcs_files\references')

# Reference files for setting gates (use your best quality/representative samples)
initial_gating_files = {
    'untreated': os.path.join(reference_folder, 'untreated.fcs'),
    'INK128': os.path.join(reference_folder, 'INK128.fcs'),
    '2iLIF': os.path.join(reference_folder, '2iLIF.fcs')
}

# Control files for fluorescence gating
thresholding_files = {
    '2iLIF': os.path.join(reference_folder, '2iLIF.fcs'),
    'Chiron': os.path.join(reference_folder, 'Chiron.fcs'),
    'N2B27': os.path.join(reference_folder, 'N2B27.fcs'),
    'untreated': os.path.join(reference_folder, 'untreated.fcs')
    }

# Output folder for gates
output_folder = reference_folder
gates_definition_path = os.path.join(output_folder, 'gate_definitions.json')

# Fluorescence channels and labels
fluor_channels = ['488nm525-40-A', '561nm610-20-A', '638nm660-10-A']
fluor_labels = ['488nm525-40 (FITC)', '561nm610-20 (PE-TexRed)', '638nm660-10 (APC)']

#%% LOAD REFERENCE DATA

print("Loading reference files for gate definition...")
reference_data = {}

for name, file_path in initial_gating_files.items():
    print(f"Loading {name} from {file_path}")
    if os.path.exists(file_path):
        try:
            sample = FCMeasurement(ID=name, datafile=file_path)
            reference_data[name] = sample.data
            print(f"{name} data shape: {reference_data[name].shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    else:
        print(f"File not found: {file_path}")

print("\nLoading control files...")
control_data = {}

for name, file_path in thresholding_files.items():
    print(f"Loading {name} from {file_path}")
    if os.path.exists(file_path):
        try:
            sample = FCMeasurement(ID=name, datafile=file_path)
            control_data[name] = sample.data
            print(f"{name} data shape: {control_data[name].shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    else:
        print(f"File not found: {file_path}")

if not reference_data:
    raise ValueError("No valid reference FCS files could be loaded!")

if not control_data:
    raise ValueError("No valid control FCS files could be loaded!")

#%% HELPER FUNCTIONS

def interactive_gate_definition(data_dict, x_channel, y_channel, title, log_scale=True):
    """Interactive polygon gate definition showing multiple samples"""
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Different colors for each sample
    colors = list(mcolors.TABLEAU_COLORS)
    
    # Epsilon for log scaling
    epsilon = 1.0
    
    # Plot each sample with different color
    for i, (sample_name, data) in enumerate(data_dict.items()):
        x = data[x_channel].values
        y = data[y_channel].values

        if log_scale:
            x_plot = np.log10(np.maximum(x, epsilon))
            y_plot = np.log10(np.maximum(y, epsilon))
        else:
            x_plot = x
            y_plot = y

        # Downsample if too many points
        if len(x_plot) > 5000:
            downsample_idx = np.random.choice(len(x_plot), size=5000, replace=False)
            x_plot_show = x_plot[downsample_idx]
            y_plot_show = y_plot[downsample_idx]
        else:
            x_plot_show = x_plot
            y_plot_show = y_plot

        # Plot
        ax.scatter(x_plot_show, y_plot_show, s=1, alpha=0.4, 
                color=colors[i % len(colors)], label=sample_name)
    
    if log_scale:
        ax.set_xlabel(f'log10({x_channel})')
        ax.set_ylabel(f'log10({y_channel})')
    else:
        ax.set_xlabel(x_channel)
        ax.set_ylabel(y_channel)
        
    ax.set_title(title + '\n(Draw polygon gate, press Done)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Create Done button
    button_ax = fig.add_axes([0.85, 0.02, 0.1, 0.06])
    done_button = Button(button_ax, 'Done', hovercolor='0.975')
    
    # Variable to store polygon vertices
    vertices = []
    gate_finished = False
    
    # PolygonSelector handler
    def onselect(verts):
        nonlocal vertices
        vertices = verts
        print(f"Gate drawn with {len(verts)} vertices")
    
    polygon_selector = PolygonSelector(ax, onselect, useblit=True)
    
    # Button callback
    def done(event):
        nonlocal gate_finished
        if len(vertices) < 3:
            print("Please select at least 3 points to form a polygon.")
            return
        
        print(f"Gate defined with {len(vertices)} vertices")
        gate_finished = True
        plt.close(fig)
    
    done_button.on_clicked(done)
    plt.tight_layout()
    plt.show()
    
    # Convert vertices to original scale if needed
    if gate_finished and len(vertices) >= 3:
        if log_scale:
            # Convert log-scale vertices to original scale
            original_vertices = [(10**vx, 10**vy) for vx, vy in vertices]
        else:
            original_vertices = vertices
        
        return original_vertices
    else:
        return None

def interactive_fluorescence_gate_definition(control_data_dict, fluor_channels, fluor_labels):
    """Define fluorescence gates using control samples"""
    
    # Colors for controls
    control_colors = {
        '2iLIF': '#FF6B6B',      # Red
        'Chiron': '#4ECDC4',     # Teal
        'N2B27': '#45B7D1',      # Blue
        'untreated': '#96CEB4'  # Green
    }
    
    # Store gates for each channel
    channel_gates = {}
    
    # Process each fluorescence channel separately
    for ch_idx, (channel, label) in enumerate(zip(fluor_channels, fluor_labels)):
        print(f"\n===== DEFINING GATE FOR {label} =====")
        
        # Create plot for this channel
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Store plot elements for toggling
        plot_elements = {}
        control_visibility = {control: True for control in control_data_dict.keys()}
        
        # Plot each control
        for control, data in control_data_dict.items():
            if len(data) == 0:
                continue
                
            # Create scatter plot
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
                               color=control_colors.get(control, '#888888'), label=control)
            plot_elements[control] = scatter
        
        # Set labels and title
        ax.set_xlabel('log10(FSC-A)')
        ax.set_ylabel(f'log10({channel})')
        ax.set_title(f'{label}\n(Toggle controls, draw polygon gate, press Done)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', markerscale=3)
        
        # Create checkbox widget for toggling controls
        checkbox_ax = fig.add_axes([0.02, 0.7, 0.15, 0.25])
        control_labels = list(control_data_dict.keys())
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
        gate_finished = False
        
        # PolygonSelector for drawing gate
        def onselect(verts):
            nonlocal gate_vertices
            gate_vertices = verts
            print(f"Gate drawn with {len(verts)} vertices")
        
        polygon_selector = PolygonSelector(ax, onselect, useblit=True)
        
        # Done button callback
        def done_callback(event):
            nonlocal gate_finished
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
                'label': label,
                'x_channel': 'FSC-A',
                'y_channel': channel
            }
            
            print(f"Gate saved for {label}")
            gate_finished = True
            plt.close(fig)
        
        done_button.on_clicked(done_callback)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.2, bottom=0.15)
        plt.show()
        
        if not gate_finished:
            print(f"No gate was drawn for {label}. Skipping this channel.")
    
    return channel_gates

#%% DEFINE GATES

# Initialize gate definitions dictionary
gate_definitions = {
    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'initial_gating_files': initial_gating_files,
    'thresholding_files': thresholding_files,
    'gates': {}
}

print("\n===== DEFINING GATE 1: CELLS (FSC-A vs SSC-A) =====")
print("This gate will identify cells from debris")

cells_vertices = interactive_gate_definition(
    reference_data, 
    'FSC-A', 'SSC-A',
    'Gate 1: Cells (FSC-A vs SSC-A)',
    log_scale=True
)

if cells_vertices:
    gate_definitions['gates']['cells'] = {
        'type': 'polygon',
        'x_channel': 'FSC-A',
        'y_channel': 'SSC-A',
        'vertices': [[float(x), float(y)] for x, y in cells_vertices],
        'log_scale': True,
        'description': 'Gate to separate cells from debris'
    }
    print("Cells gate defined successfully")
else:
    print("Cells gate definition failed")
    exit()

print("\n===== DEFINING GATE 2: SINGLETS (FSC-A vs FSC-H) =====")
print("This gate will identify single cells from doublets")

# Apply cells gate to get singlets gating data
cells_gated_data = {}
cells_path = Path(cells_vertices)

for name, data in reference_data.items():
    # Test which points are inside the cells gate
    points_to_test = np.column_stack([data['FSC-A'].values, data['SSC-A'].values])
    cells_mask = cells_path.contains_points(points_to_test)
    cells_gated_data[name] = data[cells_mask]
    print(f"{name}: {np.sum(cells_mask)} cells selected from {len(cells_mask)} events")

singlets_vertices = interactive_gate_definition(
    cells_gated_data,
    'FSC-A', 'FSC-H', 
    'Gate 2: Singlets (FSC-A vs FSC-H)',
    log_scale=True
)

if singlets_vertices:
    gate_definitions['gates']['singlets'] = {
        'type': 'polygon',
        'x_channel': 'FSC-A',
        'y_channel': 'FSC-H',
        'vertices': [[float(x), float(y)] for x, y in singlets_vertices],
        'log_scale': True,
        'description': 'Gate to separate single cells from doublets',
        'parent_gate': 'cells'
    }
    print("Singlets gate defined successfully")
else:
    print("Singlets gate definition failed")
    exit()

print("\n===== DEFINING FLUORESCENCE GATES =====")
print("Using control samples to define positive/negative boundaries")

# Apply both previous gates to control data
singlets_control_data = {}
singlets_path = Path(singlets_vertices)

for name, data in control_data.items():
    # Apply cells gate first
    points_cells = np.column_stack([data['FSC-A'].values, data['SSC-A'].values])
    cells_mask = cells_path.contains_points(points_cells)
    cells_gated = data[cells_mask]
    
    # Apply singlets gate
    points_singlets = np.column_stack([cells_gated['FSC-A'].values, cells_gated['FSC-H'].values])
    singlets_mask = singlets_path.contains_points(points_singlets)
    singlets_control_data[name] = cells_gated[singlets_mask]
    
    print(f"{name}: {len(singlets_control_data[name])} singlet cells for fluorescence gating")

# Define fluorescence gates
fluorescence_gates = interactive_fluorescence_gate_definition(
    singlets_control_data, 
    fluor_channels, 
    fluor_labels
)

if fluorescence_gates:
    for channel, gate_info in fluorescence_gates.items():
        gate_definitions['gates'][f'{channel}_positive'] = {
            'type': 'polygon',
            'x_channel': gate_info['x_channel'],
            'y_channel': gate_info['y_channel'],
            'vertices': [[float(x), float(y)] for x, y in gate_info['vertices']],
            'log_scale': True,
            'description': f'Gate for {gate_info["label"]} positive cells',
            'parent_gate': 'singlets',
            'fluorescence_channel': channel,
            'label': gate_info['label']
        }
    print(f"Fluorescence gates defined for {len(fluorescence_gates)} channels")
else:
    print("No fluorescence gates were defined")

#%% SAVE GATE DEFINITIONS

# Save gate definitions to JSON file
with open(gates_definition_path, 'w') as f:
    json.dump(gate_definitions, f, indent=2)

print(f"\n===== GATE DEFINITIONS SAVED =====")
print(f"Gate definitions saved to: {gates_definition_path}")
print(f"Total gates defined: {len(gate_definitions['gates'])}")

# Print summary
print("\nGate Summary:")
for gate_name, gate_info in gate_definitions['gates'].items():
    print(f"  {gate_name}: {gate_info['type']} gate on {gate_info['x_channel']} vs {gate_info['y_channel']}")

print("\nGate definition complete!")