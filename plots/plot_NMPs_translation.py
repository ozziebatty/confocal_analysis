import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
print("Running")

# Load data
csv_pathway = '/Users/oskar/Desktop/translation_NMPs_image_analysis/cell_types_translation_results.csv'
data = pd.read_csv(csv_pathway)

# Define columns to plot and their corresponding labels
columns_to_plot = ['Other', 'Sox1+ Bra+', 'Sox1+', 'Bra+',]
labels = ['Unlabelled', 'NMP', 'Mesoderm', 'Neural']

# Define aesthetically pleasing colours for boxes
colors = [[0.8, 0.8, 0.8],
          [1, 0.6, 0.2],
          [1, 0.3, 0.2],
          [0.6, 0.95, 0.6]]

# Create figure
plt.figure(figsize=(5, 4))  # Adjusted figure size
plt.tight_layout()

# Store boxplot data and scatter data
box_data = []
scatter_x = []
scatter_y = []
scatter_colors = []

# Define x positions
positions = np.arange(1, len(columns_to_plot) + 1)

for i, column in enumerate(columns_to_plot):
    column_data = data[column]

    # Store boxplot data
    box_data.append(column_data)

    # Scatter data points with jitter
    #x_jitter = np.random.normal(positions[i], 0.1, size=len(column_data))  # Jitter for scatter points
    scatter_x.extend(x_jitter)
    scatter_y.extend(column_data)
    scatter_colors.extend([colors[i]] * len(column_data))  # Assign colors

# Create boxplots
box = plt.boxplot(
    box_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,  # Allows coloring of the boxes
    showfliers=False,  # Remove outlier markers
    medianprops=dict(color="black", linewidth=1.5),  # Black median line
    boxprops=dict(linewidth=1.5, edgecolor='black', facecolor='none'),  # Hollow boxes with black outline
    whiskerprops=dict(linewidth=1.5, color='black'),  # Thicker whiskers
    capprops=dict(linewidth=1.5, color='black')  # Thicker caps
)

# Overlay scatter points with colors and black outlines
plt.scatter(scatter_x, scatter_y, color=scatter_colors, alpha=1.0, edgecolor='black', linewidth=0.8, zorder=2)

# Set x-axis labels
plt.grid(color='grey', linewidth=0.5, alpha=0.3)
plt.xticks(positions, labels, rotation=45)
plt.xlabel("Cell Type")
plt.ylabel("Relative Translation")
#plt.title("Cell Counts by Type")

# Custom legend for scatter colors
from matplotlib.patches import Patch
legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
#plt.legend(handles=legend_patches, title="Cell Types", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Running")

# Load data
csv_pathway = '/Users/oskar/Desktop/translation_NMPs_image_analysis/cell_types_translation_results.csv'
data = pd.read_csv(csv_pathway)

# Define columns to plot and their corresponding labels
columns_to_plot = ['Other', 'Sox1+ Bra+', 'Sox1+', 'Bra+']
labels = ['Unlabelled', 'NMP', 'Mesoderm', 'Neural']

# Define aesthetically pleasing colours for boxes
colors = [[0.8, 0.8, 0.8],
          [1, 0.6, 0.2],
          [1, 0.3, 0.2],
          [0.6, 0.95, 0.6]]

# Combine data into a list for plotting
box_data = [data[column] for column in columns_to_plot]

plt.figure(figsize=(5, 4))
#plt.tight_layout()

# First create the boxplot
box = plt.boxplot(
    box_data,
    patch_artist=True,
    widths=0.5,
    boxprops=dict(facecolor='none', linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    whis=float('inf'),  # This line makes it include all points
    capprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=1.5),
    flierprops=dict(marker='', markersize=0),
    showmeans=True,
    meanprops=dict(marker='', linewidth=1.5),
    zorder=1  # Set lower zorder for boxplot
)

# Set colors for all box elements
for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
    patch.set_facecolor('none')
    #patch.set_edgecolor(color)
    
    # Color all other elements
    #plt.setp(box['whiskers'][i*2:i*2+2], color=color)
    #plt.setp(box['caps'][i*2:i*2+2], color=color)
    plt.setp(box['medians'][i], color='black')
    plt.setp(box['means'][i], color='black')

# Then overlay scatter points with higher zorder
for i, (d, color) in enumerate(zip(box_data, colors)):
    x = np.random.normal(i + 1, 0.1, size=len(d))
    plt.scatter(x, d, color=color, alpha=1.0, edgecolor='black', linewidth=0.8, label=labels[i], zorder=2)  # Higher zorder for scatter

plt.grid(color='grey', linewidth=0.5, alpha=0.3)

plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45)
plt.xlabel("Cell Type")
plt.ylabel("Relative Translation")
#plt.ylim(80, 200)  # Adjust these numbers to your desired minimum and maximum
#plt.legend()
plt.show()