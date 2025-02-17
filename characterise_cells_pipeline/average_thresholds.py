import matplotlib.pyplot as plt
import pandas as pd

print("Running")

# Channel names (excluding DAPI)
channel_names = ["Sox1", "Sox2 Cyan", "Sox2 Orange", "Bra"]

# Load thresholds.csv into a pandas DataFrame
# Make sure thresholds.csv exists in the same directory or provide the full path
data = pd.read_csv('/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/thresholds.csv')

# Verify the expected columns exist in the DataFrame
expected_columns = ["image_name", "channel", "a", "b", "c"]
assert all(column in data.columns for column in expected_columns), "Required columns are missing from thresholds.csv"

# Filter out DAPI from the data
data = data[~data['channel'].str.contains("DAPI")]

# Initialize a dictionary to hold mean values for each channel
mean_values = {channel: [] for channel in channel_names}

# Plot separate graphs for 'a', 'b', and 'c'
metrics = ['a', 'b', 'c']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    for channel in channel_names:
        # Filter the data for the current channel
        channel_data = data[data['channel'] == channel]

        # Extract the relevant metric values
        metric_values = channel_data[metric]

        # Plot the values
        plt.scatter([channel] * len(metric_values), metric_values, label=f'{channel} - {metric}', alpha=0.6)

        # Calculate and store mean values for this channel and metric
        mean_values[channel].append(metric_values.mean())
    
    # Add graph details
    plt.title(f"Threshold values ({metric}) per channel")
    plt.xlabel("Channel")
    plt.ylabel(f"Threshold Values ({metric})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot
    plt.show()

# Print mean values
print("Mean values for each channel:")
for channel, means in mean_values.items():
    print(f"{channel}: a = {means[0]:}, b = {means[1]:}, c = {means[2]:}")