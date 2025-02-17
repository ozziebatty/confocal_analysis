import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/correlations.csv'
df = pd.read_csv(file_path)

# Step 1: Filter out DAPI comparisons
df = df[(df['Channel 1'] != 'DAPI') & (df['Channel 2'] != 'DAPI')]


# Step 2: Transform the Data
# Create pivot tables for Pearson Correlation and Manders Overlap
df_pearson = df.pivot_table(index='Z-slice', columns=['Channel 1', 'Channel 2'], values='Pearson Correlation Value')
df_manders = df.pivot_table(index='Z-slice', columns=['Channel 1', 'Channel 2'], values='Manders Overlap Value')

# Flatten the MultiIndex columns
df_pearson.columns = [f'{ch1}:{ch2}' for ch1, ch2 in df_pearson.columns]
df_manders.columns = [f'{ch1}:{ch2}' for ch1, ch2 in df_manders.columns]

df_pearson.reset_index(inplace=True)
df_manders.reset_index(inplace=True)

# Step 3: Plot the Data
fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot Pearson Correlation Values
for column in df_pearson.columns[1:]:
    ax[0].plot(df_pearson['Z-slice'], df_pearson[column], marker='o', label=column)

ax[0].set_xlabel('Z-slice')
ax[0].set_ylabel('Pearson Correlation Value')
ax[0].set_title('Pearson Correlation Across Z-slices')
ax[0].legend()
ax[0].grid(True)

# Plot Manders Overlap Values
for column in df_manders.columns[1:]:
    ax[1].plot(df_manders['Z-slice'], df_manders[column], marker='o', label=column)

ax[1].set_xlabel('Z-slice')
ax[1].set_title('Manders Overlap Across Z-slices')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
