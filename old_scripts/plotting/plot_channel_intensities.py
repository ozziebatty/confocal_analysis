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

#Remove NaNs and infinities
df = df.dropna(subset=['channel_2_norm', 'channel_3_norm'])
df = df[~np.isinf(df['channel_2_norm']) & ~np.isinf(df['channel_3_norm'])]

# Define color mapping based on the 'fate' column
color_map = {
    'Bra': 'red',
    'Sox2': 'green',
    'Bra_Sox2': 'purple',
    'unlabelled': 'black'
}

# Apply the color mapping to a new column
df['color'] = df['fate'].map(color_map)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['channel_2_norm'], df['channel_3_norm'], alpha=0.7, c=df['color'], s=10)

### Calculate the line of best fit
##coefficients = np.polyfit(df['channel_2_norm'], df['channel_3_norm'], deg=1)
##poly = np.poly1d(coefficients)
##best_fit_line = poly(df['channel_2_norm'])
##
### Plot the best fit line
##plt.plot(df['channel_2_norm'], best_fit_line, alpha=0.7)

# Customize the plot
plt.title('Scatter Plot of Channel 2 vs Channel 3')
plt.xlabel('Channel 1')
plt.ylabel('Channel 2')
plt.grid(True)

# Show the plot
plt.show()
