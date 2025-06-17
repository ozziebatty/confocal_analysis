#%% IMPORT PACKAGES

print("Running Gate Application Script")
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from matplotlib.path import Path
import matplotlib.colors as mcolors

#%% CONFIGURE SETTINGS

# ========== Configure settings ==========
# Base folder containing the reference gate definitions
reference_folder = os.path.normpath(r'Y:\Room225_SharedFolder\CytoflexLX_data\oskar\relevant_fcs_files\references')

# Base folder containing FCS files to analyze
sample_folder = os.path.normpath(r'Y:\Room225_SharedFolder\CytoflexLX_data\oskar\relevant_fcs_files\d2_treatments\B')

# All sample files to analyze (can include multiple replicates, treatments, etc.)
sample_files = {
    'untreated': os.path.join(sample_folder, 'untreated.fcs'),
    'BMH21': os.path.join(sample_folder, 'BMH21.fcs'),
    'INK128': os.path.join(sample_folder, 'INK128.fcs'),
    'Sal003': os.path.join(sample_folder, 'Sal003.fcs'),
    'MHY1485': os.path.join(sample_folder, 'MHY1485.fcs'),
    'puromycin': os.path.join(sample_folder, 'puromycin.fcs')
}

# For multiple replicates/experiments, you can organize like this:
# sample_files = {
#     'Exp1_untreated_rep1': os.path.join(sample_folder, 'Exp1', 'untreated_rep1.fcs'),
#     'Exp1_untreated_rep2': os.path.join(sample_folder, 'Exp1', 'untreated_rep2.fcs'),
#     'Exp1_treatment_rep1': os.path.join(sample_folder, 'Exp1', 'treatment_rep1.fcs'),
#     'Exp2_untreated_rep1': os.path.join(sample_folder, 'Exp2', 'untreated_rep1.fcs'),
#     # ... etc
# }

# Output folder (will save to sample folder)
output_folder = sample_folder

# Path to gate definitions (from reference folder)
gates_definition_path = os.path.join(reference_folder, 'gate_definitions.json')

# Output paths
results_csv_path = os.path.join(output_folder, 'gating_results.csv')
consolidated_csv_path = os.path.join(output_folder, 'consolidated_cell_data.csv')
analysis_summary_path = os.path.join(output_folder, 'analysis_summary.json')

#%% LOAD GATE DEFINITIONS

print("Loading gate definitions...")
if not os.path.exists(gates_definition_path):
    raise FileNotFoundError(f"Gate definitions file not found: {gates_definition_path}")
    
with open(gates_definition_path, 'r') as f:
    gate_definitions = json.load(f)

print(f"Loaded gate definitions created on: {gate_definitions['created_date']}")
print(f"Available gates: {list(gate_definitions['gates'].keys())}")

#%% LOAD SAMPLE DATA

print("\nLoading sample files...")
sample_data = {}

for name, file_path in sample_files.items():
    print(f"Loading {name} from {file_path}")
    if os.path.exists(file_path):
        try:
            sample = FCMeasurement(ID=name, datafile=file_path)
            sample_data[name] = sample.data
            print(f"  {name} data shape: {sample_data[name].shape}")
            print(f"  FSC-A range: {sample_data[name]['FSC-A'].min():.0f} to {sample_data[name]['FSC-A'].max():.0f}")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    else:
        print(f"  File not found: {file_path}")

if not sample_data:
    raise ValueError("No valid sample FCS files could be loaded!")

#%% HELPER FUNCTIONS

def apply_polygon_gate(data, gate_info):
    """Apply a polygon gate to data"""
    vertices = gate_info['vertices']
    x_channel = gate_info['x_channel']
    y_channel = gate_info['y_channel']
    
    # Create Path object from vertices
    gate_path = Path(vertices)
    
    # Get the points to test
    points_to_test = np.column_stack([data[x_channel].values, data[y_channel].values])
    
    # Test which points are inside the gate
    mask = gate_path.contains_points(points_to_test)
    
    return mask, data[mask]

#%% APPLY GATES TO ALL SAMPLES

print("\n===== APPLYING GATES TO ALL SAMPLES =====")

# Store results for all samples
all_results = {}
comparison_data = []
consolidated_data = []

for sample_name, data in sample_data.items():
    print(f"\nProcessing {sample_name}...")
    
    # Dictionary to store gating results for this sample
    sample_results = {}
    current_data = data.copy()
    
    # Apply gates in sequence
    gate_sequence = ['cells', 'singlets']
    
    for gate_name in gate_sequence:
        if gate_name in gate_definitions['gates']:
            print(f"  Applying {gate_name} gate...")
            gate_info = gate_definitions['gates'][gate_name]
            
            # Apply gate
            mask, gated_data = apply_polygon_gate(current_data, gate_info)
            
            # Store results
            sample_results[gate_name] = {
                'mask': mask,
                'data': gated_data,
                'count': int(np.sum(mask)),
                'total': int(len(mask)),
                'percentage': float(100 * np.sum(mask) / len(mask)) if len(mask) > 0 else 0
            }
            
            print(f"    {gate_name}: {sample_results[gate_name]['count']} / {sample_results[gate_name]['total']} ({sample_results[gate_name]['percentage']:.2f}%)")
            
            # Update current_data for next gate in sequence
            current_data = gated_data
            
            # Save gated cells
            output_path = os.path.join(output_folder, f'{sample_name}_{gate_name}.csv')
            gated_data.to_csv(output_path, index=False)
            
            # Add to comparison data
            comparison_data.append({
                'Treatment': sample_name,
                'Gate': gate_name,
                'Channel': gate_info.get('fluorescence_channel', ''),
                'Label': gate_info.get('label', gate_name),
                'Count_Positive': sample_results[gate_name]['count'],
                'Total_Events': sample_results[gate_name]['total'],
                'Percentage_Positive': sample_results[gate_name]['percentage']
            })
    
    # Apply fluorescence gates to singlets
    if 'singlets' in sample_results:
        singlets_data = sample_results['singlets']['data']
        
        for gate_name in gate_definitions['gates'].keys():
            if 'positive' in gate_name:
                print(f"  Applying {gate_name} gate to singlets...")
                gate_info = gate_definitions['gates'][gate_name]
                
                # Apply fluorescence gate
                mask, positive_cells = apply_polygon_gate(singlets_data, gate_info)
                
                # Store results
                sample_results[gate_name] = {
                    'mask_in_singlets': mask,
                    'data': positive_cells,
                    'count': int(np.sum(mask)),
                    'total': int(len(singlets_data)),
                    'percentage': float(100 * np.sum(mask) / len(singlets_data)) if len(singlets_data) > 0 else 0
                }
                
                print(f"    {gate_name}: {sample_results[gate_name]['count']} / {sample_results[gate_name]['total']} ({sample_results[gate_name]['percentage']:.2f}%)")
                
                # Save fluorescence-positive cells
                output_path = os.path.join(output_folder, f'{sample_name}_{gate_name}.csv')
                positive_cells.to_csv(output_path, index=False)
                
                # Add to comparison data
                comparison_data.append({
                    'Treatment': sample_name,
                    'Gate': gate_name,
                    'Channel': gate_info.get('fluorescence_channel', ''),
                    'Label': gate_info.get('label', gate_name),
                    'Count_Positive': sample_results[gate_name]['count'],
                    'Total_Events': sample_results[gate_name]['total'],
                    'Percentage_Positive': sample_results[gate_name]['percentage']
                })
    
    # Store results for this sample
    all_results[sample_name] = sample_results
    
    # Create consolidated dataset for this sample
    if 'singlets' in sample_results:
        singlets = sample_results['singlets']['data']
        
        # Get fluorescence status for each singlet cell
        fluor_status = {}
        for gate_name in gate_definitions['gates'].keys():
            if 'positive' in gate_name and gate_name in sample_results:
                fluor_status[gate_name] = sample_results[gate_name]['mask_in_singlets']
            elif 'positive' in gate_name:
                # If gate wasn't applied, mark all as negative
                fluor_status[gate_name] = np.zeros(len(singlets), dtype=bool)
        
        # Add each singlet cell to consolidated data
        for idx in range(len(singlets)):
            cell_data = {
                'Treatment': sample_name,
                'Cell_ID': f"{sample_name}_{idx}",
                'FSC-A': singlets['FSC-A'].iloc[idx],
                'SSC-A': singlets['SSC-A'].iloc[idx],
                'FSC-H': singlets['FSC-H'].iloc[idx]
            }
            
            # Add fluorescence channels
            for channel in ['488nm525-40-A', '561nm610-20-A', '638nm660-10-A']:
                if channel in singlets.columns:
                    cell_data[channel] = singlets[channel].iloc[idx]
            
            # Add fluorescence status
            for gate_name in fluor_status.keys():
                status_name = gate_name.replace('_positive', '_pos')
                cell_data[status_name] = fluor_status[gate_name][idx] if idx < len(fluor_status[gate_name]) else False
            
            consolidated_data.append(cell_data)

#%% SAVE RESULTS

print("\n===== SAVING RESULTS =====")

# Save comparison data
if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(results_csv_path, index=False)
    print(f"Gating results saved to: {results_csv_path}")

# Save consolidated dataset
if consolidated_data:
    consolidated_df = pd.DataFrame(consolidated_data)
    consolidated_df.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated cell data saved to: {consolidated_csv_path}")
    print(f"Total cells in consolidated dataset: {len(consolidated_df)}")

# Create analysis summary
analysis_summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'reference_folder': reference_folder,
    'sample_folder': sample_folder,
    'gate_definitions_file': gates_definition_path,
    'gate_definitions_date': gate_definitions['created_date'],
    'samples_analyzed': list(sample_data.keys()),
    'gates_applied': list(gate_definitions['gates'].keys()),
    'total_samples': len(sample_data),
    'total_singlet_cells': len(consolidated_data) if consolidated_data else 0,
    'sample_summary': {}
}

# Add per-sample summary
for sample_name, results in all_results.items():
    sample_summary = {
        'total_events': len(sample_data[sample_name]),
        'gating_summary': {}
    }
    
    for gate_name, gate_results in results.items():
        sample_summary['gating_summary'][gate_name] = {
            'count': gate_results['count'],
            'total': gate_results['total'],
            'percentage': gate_results['percentage']
        }
    
    analysis_summary['sample_summary'][sample_name] = sample_summary

# Save analysis summary
with open(analysis_summary_path, 'w') as f:
    json.dump(analysis_summary, f, indent=2)
print(f"Analysis summary saved to: {analysis_summary_path}")

#%% CREATE FINAL SUMMARY PLOT

print("\n===== CREATING FINAL SUMMARY PLOT =====")

# Simple plot showing the two main gates across treatments
main_gates = ['cells', 'singlets']
n_points = 5000  # Number of points to show per sample

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for gate_idx, gate_name in enumerate(main_gates):
    if gate_name not in gate_definitions['gates']:
        continue
        
    ax = axes[gate_idx]
    gate_info = gate_definitions['gates'][gate_name]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for sample_idx, (sample_name, data) in enumerate(sample_data.items()):
        # Subsample data
        if len(data) > n_points:
            plot_data = data.sample(n=n_points, random_state=42)
        else:
            plot_data = data
        
        # Get plot coordinates
        x_data = plot_data[gate_info['x_channel']].values
        y_data = plot_data[gate_info['y_channel']].values
        
        # Apply log scale if needed
        if gate_info.get('log_scale', False):
            x_plot = np.log10(np.maximum(x_data, 1.0))
            y_plot = np.log10(np.maximum(y_data, 1.0))
            xlabel = f"log10({gate_info['x_channel']})"
            ylabel = f"log10({gate_info['y_channel']})"
        else:
            x_plot = x_data
            y_plot = y_data
            xlabel = gate_info['x_channel']
            ylabel = gate_info['y_channel']
        
        # Simple scatter plot
        color = colors[sample_idx % len(colors)]
        ax.scatter(x_plot, y_plot, c=color, s=0.5, alpha=0.5, 
                  label=sample_name if gate_idx == 0 else "")
    
    # Draw gate boundary
    vertices = gate_info['vertices']
    if gate_info.get('log_scale', False):
        plot_vertices = [(np.log10(max(vx, 1.0)), np.log10(max(vy, 1.0))) 
                        for vx, vy in vertices]
    else:
        plot_vertices = vertices
    
    plot_vertices.append(plot_vertices[0])  # Close polygon
    gate_x, gate_y = zip(*plot_vertices)
    ax.plot(gate_x, gate_y, 'black', linewidth=2)
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{gate_name.title()} Gate')
    ax.grid(True, alpha=0.3)

# Legend on first plot only
axes[0].legend()

plt.suptitle('Gating Results Across Treatments', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'final_gating_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Results saved to: {output_folder}")
print(f"Final plot saved as: final_gating_summary.png")