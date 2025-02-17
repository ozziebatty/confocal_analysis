import matplotlib.pyplot as plt
import numpy as np

print("Running")

# Data for the last three columns
sox2_plus = [150.871331, 150.249326, 91.891491, 105.525445, 113.369511, 139.667364]
unmarked = [156.900962, 178.918992, 179.080346, 175.900942, 165.937203, 152.334682]
oct4_sox2 = [131.318182, 159.739768, 94.401066, 126.184801, 131.551202, 139.458333]

# Combine data into a list for plotting
#data = [sox2_plus, oct4_sox2_plus, unmarked]
#labels = ['Sox2+', 'Oct4+ Sox2+', 'Unmarked']
#colors = ['green', 'red', 'grey']

data = [unmarked, oct4_sox2]
labels = ['Unmarked', 'Pluripotent\n(Oct4+, Sox2+)']
colors = [[0.4, 0.4, 0.4], [0.3, 0.7, 0.85]]
x_positions = [1, 2]

plt.figure(figsize=(2.5, 4))
#plt.tight_layout()

# First create the boxplot
box = plt.boxplot(
    data,
    #labels=labels,
    patch_artist=True,
    widths = 0.5,
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
for i, (d, color) in enumerate(zip(data, colors)):
    #x = np.random.normal(i + 1, 0.1, size=len(d))
    x = np.random.normal(x_positions[i], 0.1, size=len(d))
    plt.scatter(x, d, color=color, alpha=1.0, edgecolor='black', linewidth=0.8, label=labels[i], zorder=2)  # Higher zorder for scatter

plt.grid(color='grey', linewidth=0.5, alpha=0.3)

plt.subplots_adjust(left=0.3)

plt.ylabel('Translation (AU)')
plt.ylim(80, 200)  # Adjust these numbers to your desired minimum and maximum
#plt.legend()
plt.show()