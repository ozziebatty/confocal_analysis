import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path to your CSV file
file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/gastruloid_z_characterised_cells.csv'

# Read the data into a DataFrame
df = pd.read_csv(file_path)

# Print the column names to verify
print("Columns in DataFrame:", df.columns)

# Define colors for each fate
color_map = {
    'Bra': 'red',
    'Sox2': 'green',
    'Bra_Sox2': 'purple',
    'Unlabelled': 'blue'
}

# Create a figure and axis
fig, ax = plt.subplots()

# Check if the required columns are present
if 'fate' in df.columns and 'pixel_count' in df.columns:
    print("\nStatistics by Fate:")
    for fate in color_map.keys():
        subset = df[df['fate'] == fate]
        mean_pixel_count = subset['pixel_count'].mean()
        std_pixel_count = subset['pixel_count'].std()
        volume = subset['pixel_count'] * 1.406
        mean_volume = volume.mean()
        std_volume = volume.std()
        print(f"Fate: {fate} | Mean: {mean_pixel_count:.2f} | Std Dev: {std_pixel_count:.2f}")


    # Convert 'fate' to a categorical type with numerical codes
    df['fate_code'] = df['fate'].astype('category').cat.codes
    categories = df['fate'].astype('category').cat.categories

    # Plot each fate with the specified color
    for fate, color in color_map.items():
        subset = df[df['fate'] == fate]
        
        # Add jitter to numerical x-values
        jitter = np.random.normal(0, 0.1, size=len(subset))
        x_values = subset['fate_code'] + jitter

        # Scatter plot
        ax.scatter(x_values, (subset['pixel_count'] * 1.406), color=color, label=fate, s=0.1, alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Fate')
    ax.set_ylabel('Volume in µm3')
    ax.set_title('Pixel Count by Fate')
    ax.legend(title='Fate')

    # Set x-ticks to the categorical values
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45)  # Rotate x labels for better readability

    # Display the plot
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()
else:
    print("Required columns are missing from the DataFrame.")
