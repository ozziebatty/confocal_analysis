import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Running")

"Reads in characterised_cells, plots total number of cells by treatment as boxplots"

# Load data
csv_pathway = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/characterised_cells_summary_by_cell_type.csv'
data = pd.read_csv(csv_pathway)

# Define aesthetically pleasing colours for boxes
labels = ['Control', 'Treatment']
colors = [[0.8, 0.8, 0.8], [0.6, 0.6, 0.6]]

# Create figure
plt.figure(figsize=(6, 6))  # Adjusted figure size

# Extract total number of cells by treatment
control_data = data[data['condition'] == 'control']['Total']
treatment_data = data[data['condition'] == 'treatment']['Total']

# Store boxplot data
box_data = [control_data, treatment_data]

# Define x positions
positions = [1, 2]

# Scatter data points with jitter
x_control = np.random.normal(positions[0], 0.1, size=len(control_data))  # Jitter for control
x_treatment = np.random.normal(positions[1], 0.1, size=len(treatment_data))  # Jitter for treatment

scatter_x = np.concatenate([x_control, x_treatment])
scatter_y = np.concatenate([control_data, treatment_data])
scatter_colors = [colors[0]] * len(control_data) + [colors[1]] * len(treatment_data)  # Assign colors

# Create boxplots
box = plt.boxplot(
    box_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,  # Allows coloring of the boxes
    showfliers=False,  # Remove outlier markers
    medianprops=dict(color="black", linewidth=2),  # Black median line
    boxprops=dict(linewidth=2, edgecolor='black', facecolor='none'),  # Hollow boxes with black outline
    whiskerprops=dict(linewidth=2, color='black'),  # Thicker whiskers
    capprops=dict(linewidth=2, color='black')  # Thicker caps
)

# Overlay scatter points with colors and black outlines
plt.scatter(scatter_x, scatter_y, color=scatter_colors, alpha=1.0, edgecolor='black', linewidth=0.8, zorder=2)

# Set x-axis labels
plt.grid(color='grey', linewidth=0.5, alpha=0.3)
plt.xticks(positions, labels, rotation=45)
plt.xlabel("Condition")
plt.ylabel("Total Number of Cells")
plt.title("Total Number of Cells by Treatment")

# Custom legend for scatter colors
from matplotlib.patches import Patch
legend_patches = [Patch(color=colors[0], label='Control'), Patch(color=colors[1], label='Treatment')]
plt.legend(handles=legend_patches, title="Condition", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()