import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print("Running")

# Load data
csv_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/characterised_cells_summary_by_cell_type.csv'
data = pd.read_csv(csv_pathway)

# Calculate proportions for each channel
channels = [col for col in data.columns if col not in ['condition', 'Unlabelled', 'Total', 'image_name']]
for channel in channels:
    data[f"{channel}_proportion"] = data[channel] / data['Total']

# Define colours for boxes
labels = ['Pluripotent', 'NMP', 'Mesoderm', 'Neural']

colors = [[0.5, 0.9, 1], [0.375, 0.675, 0.75],
          [1, 0.6, 0.2], [0.75, 0.45, 0.15],
          [1, 0.3, 0.2], [0.75, 0.225, 0.15],
          [0.6, 0.95, 0.6], [0.45, 0.713, 0.45]]

# Create figure
plt.figure(figsize=(8.5, 6))

positions = []
box_data = []
scatter_x = []
scatter_y = []
scatter_colors = []
scatter_markers = []  # Store markers for control and treatment

# Define legend colours for control and treatment
control_marker_color = "#D3D3D3"  # Light grey for legend
treatment_marker_color = "#7D7D7D"  # Dark grey for legend

for i, channel in enumerate(channels):
    control_data = data[data['condition'] == 'control'][f"{channel}_proportion"]
    treatment_data = data[data['condition'] == 'treatment'][f"{channel}_proportion"]

    # Store boxplot data
    box_data.append(control_data)
    box_data.append(treatment_data)

    # Define x positions
    control_pos = i * 3 + 1
    treatment_pos = i * 3 + 2
    positions.extend([control_pos, treatment_pos])

    # Scatter data points with jitter
    x_control = np.random.normal(control_pos, 0.15, size=len(control_data))
    x_treatment = np.random.normal(treatment_pos, 0.15, size=len(treatment_data))

    scatter_x.extend(x_control)
    scatter_x.extend(x_treatment)
    scatter_y.extend(control_data)
    scatter_y.extend(treatment_data)

    # Assign colors and markers (keeping original scatter colours)
    scatter_colors.extend([colors[i * 2]] * len(control_data))  # Control points
    scatter_colors.extend([colors[i * 2 + 1]] * len(treatment_data))  # Treatment points
    scatter_markers.extend(['o'] * len(control_data))  # Circles for control
    scatter_markers.extend(['s'] * len(treatment_data))  # Squares for treatment

# Create boxplots
plt.boxplot(
    box_data,
    positions=positions,
    widths=0.75,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.5, edgecolor='black', facecolor='none'),
    whiskerprops=dict(linewidth=1.5, color='black'),
    capprops=dict(linewidth=1.5, color='black')
)

# Overlay scatter points (maintaining original colours)
for x, y, color, marker in zip(scatter_x, scatter_y, scatter_colors, scatter_markers):
    plt.scatter(x, y, color=color, alpha=1.0, edgecolor='black', linewidth=0.8, marker=marker, zorder=2)

# Set x-axis labels
plt.grid(color='grey', linewidth=0.5, alpha=0.3)
plt.xticks([i * 3 + 1.5 for i in range(len(channels))], channels, rotation=45)
plt.xlabel("Channel")
plt.ylabel("Proportion of total cells")
plt.ylim(-0.005, 0.08)

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=control_marker_color, markersize=10, label='Control', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=treatment_marker_color, markersize=10, label='BMH21 60uM', markeredgecolor='black')
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
