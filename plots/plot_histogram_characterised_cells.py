#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load data
file_path = "/Users/oskar/Desktop/format_test/SBSO_stellaris_data/characterised_cells.csv"
df = pd.read_csv(file_path)  # Assuming tab-separated from your headings
print(df.columns)

# List of channels to compare against channel_3
channels = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']
channel_axes = [4, 2, 4, 15, 15]
channel_names = ['DAPI', 'Sox1', 'Bra', 'OPP', 'Sox2']
y_channel = 'channel_3'
x_channels = [ch for ch in channels if ch != y_channel]

#%%
# Plot settings
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

for i, ch in enumerate(x_channels):
    sns.histplot(
        data=df,
        x=ch,
        y=y_channel,
        bins=100,
        pthresh=0.1,
        cmap="mako",
        ax=axes[i]
    )
    axes[i].set_title(f'{ch} vs channel_3')
    axes[i].set_xlim(0, channel_axes[i])
    axes[i].set_ylim(0, 15)
    axes[i].set_xlabel(ch)
    axes[i].set_ylabel('channel_3')

plt.tight_layout()
#plt.show()

# %%
#%%
# Plot channel_3 values for top 20% signal cells from each other channel
#%%
# Create a long-form dataframe: rows = cells, columns = group (based on top 20% in channel) and channel_3 value

threshold_percentile = 99.9
channels_to_filter = ['channel_0', 'channel_1', 'channel_2', 'channel_4']
y_channel = 'channel_3'

# Collect top 20% cells per channel, tag them with channel name
subset_list = []

for ch in channels_to_filter:
    threshold = df[ch].quantile(threshold_percentile / 100)
    top_cells = df[df[ch] >= threshold].copy()
    top_cells['group'] = ch  # label by the channel used
    subset_list.append(top_cells[['group', y_channel]])

# Combine into one dataframe
plot_df = pd.concat(subset_list, ignore_index=True)

# Plot as boxplot (or change to stripplot/swarmplot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=plot_df, x='group', y=y_channel, palette='mako')

# Optional: add jittered dots on top for detail
sns.stripplot(data=plot_df, x='group', y=y_channel, color='black', size=3, jitter=True, alpha=0.5)

plt.title(f'{y_channel} signal in top 20% cells from other channels')
plt.ylabel(y_channel)
plt.xlabel('Top 20% group based on channel')
plt.ylim(0, 25)
plt.tight_layout()
plt.show()

# %%
