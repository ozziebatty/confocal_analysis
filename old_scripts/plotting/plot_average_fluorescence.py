import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/normalized_average_fluorescence.csv'
average_fluorescence = pd.read_csv(file_path)

# Create a new column for Z slices if it doesn't exist
average_fluorescence['z'] = average_fluorescence.index

colors = {
    'DAPI': 'blue',
    'Bra': 'red',
    'Sox2': 'lightgreen',
    'OPP': 'cyan'
}


# Create a new DataFrame to store the normalized data
normalized_average_fluorescence = average_fluorescence.copy()
for column in average_fluorescence.columns:
    if column != 'z':  # Skip the Z slice column
        max_value = average_fluorescence[column].max()
        if max_value != 0:  # Prevent division by zero
            normalized_average_fluorescence[column] = average_fluorescence[column] / max_value

df_to_plot = average_fluorescence

plt.figure(figsize=(10, 6))

for column in df_to_plot.columns:
    if column != 'z':  # Skip the Z slice column
        plt.plot(df_to_plot['z'], df_to_plot[column], marker='o', label=column, color=colors.get(column, 'black'))

# Customize the plot
plt.title('Average Fluorescence Across Z Slices')
plt.xlabel('Z Slice')
plt.ylabel('Average Fluorescence')
plt.legend()
plt.grid(True)

plt.show()
