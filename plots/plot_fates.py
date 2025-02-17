import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# New values for results_df
results_df = pd.DataFrame({
    'Purple': [0.547109, 0.827710, 0.608043, 0.737190, 0.505297, 0.846662, 0.783123],
    'Red': [0.639704, 0.899738, 0.671538, 0.653030, 0.524778, 0.801999, 0.709728],
    'Green': [0.768022, 0.856243, 0.708143, 0.843565, 0.675713, 0.700869, 0.843916],
    'Blue': [0.887903, 1.018669, 0.939213, 0.694260, 0.879438, 0.863230, 0.786952]
})

# Rename columns with the new category names
results_df.columns = ['Sox1+ Bra+', 'Bra+', 'Sox1+', 'Other']

# Reorder columns to match the specified x-axis order
results_df = results_df[['Other', 'Bra+', 'Sox1+', 'Sox1+ Bra+']]

# Define colors for each category, with Grey for "Other" and Orange for "Sox1+ Bra+"
colors = {'Sox1+ Bra+': 'orange', 'Bra+': 'red', 'Sox1+': 'green', 'Other': 'grey'}

# Calculate mean and standard deviation for each category
means = results_df.mean()
std_devs = results_df.std()

# Plotting
plt.figure(figsize=(6, 6))

# Plot individual data points with category-specific colors
for category in results_df.columns:
    plt.scatter(
        [category] * len(results_df),
        results_df[category],
        alpha=1.0,
        s=35,
        color=colors[category],
        label=category if category not in plt.gca().get_legend_handles_labels()[1] else ""
    )

# Plot means with error bars
plt.errorbar(
    results_df.columns, 
    means, 
    yerr=std_devs, 
    fmt='o', 
    color='black', 
    alpha = 0.5,
    capsize=5
)

# Labels, title, and legend
plt.xlabel('Cell Category')
plt.ylabel('OPP (Average Pixel Value)')
plt.title('OPP Values by Cell Category across Images')
plt.legend(title="Category")
plt.show()
