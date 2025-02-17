import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path to your CSV file
file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/gastruloid_z_characterised_cells.csv'

# Read the data into a DataFrame
df = pd.read_csv(file_path)

# Normalize channels 2-4 relative to channel_1
df['channel_1_norm'] = df['channel_1'] / df['channel_0']
df['channel_2_norm'] = df['channel_2'] / df['channel_0']
df['channel_3_norm'] = df['channel_3'] / df['channel_0']

# Remove NaNs and infinities
df = df.dropna(subset=['channel_2_norm', 'channel_3_norm'])
df = df[~np.isinf(df['channel_2_norm']) & ~np.isinf(df['channel_3_norm'])]

# Create a boxplot with 'fate' on the x-axis and 'channel_3_norm' on the y-axis
plt.figure(figsize=(10, 6))  # Optional: Set figure size
df.boxplot(column='pixel_count', by='fate', grid=False, showfliers=False)

# Customize the plot
plt.title('Boxplot of Channel 3 by Fate')
plt.suptitle('')  # Suppress the default title to avoid redundancy
plt.xlabel('Fate')
plt.ylabel('Channel 3 (Normalized)')
plt.grid(False)

# Show the plot
plt.show()
