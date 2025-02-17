import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the folder path
folder_path = "/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/SBSE_BMH21_analysis"

# Initialize an empty DataFrame to store data from all files
combined_data = pd.DataFrame()

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Assuming files are in CSV format
        file_path = os.path.join(folder_path, file_name)
        # Read the data and append it to the combined DataFrame
        file_data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, file_data], ignore_index=True)

# Ensure all necessary columns are present
required_columns = [
    "name", "condition", "channel",
    "relative_pixel_intensity", "channel_intensity", 
    "pixels_over_threshold", "relative_channel_intensity"
]
if not all(col in combined_data.columns for col in required_columns):
    raise ValueError("One or more required columns are missing in the data files.")

# Define the metrics to compare
metrics = [
    "relative_pixel_intensity", "channel_intensity",
    "pixels_over_threshold", "relative_channel_intensity"
]

# Function to plot a batch of 8 boxplots
def plot_batch(batch_number, metrics, combined_data, output_folder):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns for 8 plots
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot the 8 boxplots
    for idx, (metric, channel) in enumerate([(m, c) for m in metrics for c in range(4)][batch_number * 8:(batch_number + 1) * 8]):
        ax = axes[idx]
        
        # Filter data for the current channel and metric
        data = combined_data[combined_data["channel"] == channel]
        
        # Prepare data for boxplot
        control_data = data[data["condition"] == "control"][metric].dropna()
        treated_data = data[data["condition"] == "treated"][metric].dropna()
        
        # Create the boxplot
        box = ax.boxplot(
            [control_data, treated_data],
            labels=["control", "treated"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red")
        )
        
        # Overlay points for individual data
        for i, group_data in enumerate([control_data, treated_data]):
            x_vals = np.random.normal(i + 1, 0.04, size=len(group_data))  # Add jitter for clarity
            ax.scatter(x_vals, group_data, alpha=0.6, color="black", s=10)

        ax.set_title(f"Channel {channel} - {metric}")
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric)

    # Hide unused subplots (if fewer than 8 plots)
    for idx in range(len([(m, c) for m in metrics for c in range(4)][batch_number * 8:(batch_number + 1) * 8]), len(axes)):
        axes[idx].axis("off")
    
    # Save the figure
    output_file = os.path.join(output_folder, f"boxplots_batch_{batch_number + 1}.png")
    plt.savefig(output_file)
    print(f"Boxplots batch {batch_number + 1} saved to {output_file}")
    plt.show()

# Create output folder for saving plots
output_folder = "/Users/oskar/Desktop/steventon_lab/image_analysis"
os.makedirs(output_folder, exist_ok=True)

# Plot two batches of 8 boxplots each
plot_batch(0, metrics, combined_data, output_folder)  # First batch
plot_batch(1, metrics, combined_data, output_folder)  # Second batch
