import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === SETTINGS ===
days_to_plot = '2-5'  # Set the days variable to the desired value
csv_path = r"C:\Users\ob361\data\glucose_Bra_Sox2\masked_channel_intensity_means.csv"

# === LOAD DATA ===
#df_unfiltered = pd.read_csv(csv_path)
#df = df_unfiltered[df_unfiltered['days'] == days_to_plot]
df = pd.read_csv(csv_path)

# Ensure correct dtypes
df['treatment'] = df['treatment'].astype(str)
df['replicate'] = df['replicate'].astype(str)
df['days'] = df['days'].astype(str)
df['set'] = df['set'].astype(str)

# Melt for absolute signal
signal_cols = [f'channel_{i}' for i in range(4)]
df_long = df.melt(
    id_vars=['image_name', 'inverse', 'treatment', 'set', 'replicate'],
    value_vars=signal_cols,
    var_name='channel',
    value_name='signal'
)

# Colour palette and channel order
channel_order = ['channel_0', 'channel_1', 'channel_2', 'channel_3']
colour_palette = {
    'channel_0': 'grey',
    'channel_1': 'green',
    'channel_2': 'red',
    'channel_3': 'pink'
}


# === ABSOLUTE SIGNAL — side-by-side with shared y-axis ===
unique_sets = sorted(df_long['set'].unique())
n_sets = len(unique_sets)
unique_days = sorted(df_long['days'].unique())
n_days = len(unique_days)
print(f"Unique sets: {unique_sets}")
print(f"Number of sets: {n_sets}")

fig, axes = plt.subplots(n_days, n_sets, figsize=(6 * n_sets, 6*n_days), sharey=True)

for i, this_set in enumerate(unique_sets):
    subset = df_long[df_long['set'] == this_set]
    ax = axes[i] if n_sets > 1 else axes

    sns.boxplot(data=subset, x='channel', y='signal', hue='treatment',
                order=channel_order, ax=ax)
    sns.stripplot(data=subset, x='channel', y='signal', hue='treatment',
                  dodge=True, jitter=True, linewidth=0.5, alpha=0.6,
                  order=channel_order, ax=ax)

    ax.set_title(f"Absolute signal — {this_set}")
    ax.set_xlabel("Channel")
    if i == 0:
        ax.set_ylabel("Signal")
    else:
        ax.set_ylabel("")
    ax.legend_.remove()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()
plt.show()

# === RELATIVE SIGNAL CALCULATION ===
rel_df = df.copy()
for i in range(1, 4):
    rel_df[f'rel_channel_{i}'] = rel_df[f'channel_{i}'] / rel_df['channel_0']

rel_cols = [f'rel_channel_{i}' for i in range(1, 4)]
rel_long = rel_df.melt(
    id_vars=['image_name', 'inverse', 'treatment', 'set', 'replicate'],
    value_vars=rel_cols,
    var_name='channel',
    value_name='relative_signal'
)
rel_long['channel'] = rel_long['channel'].str.replace('rel_channel_', 'channel_', regex=False)

# === RELATIVE SIGNAL — side-by-side with shared y-axis ===
unique_sets = sorted(rel_long['set'].unique())
n_sets = len(unique_sets)

fig, axes = plt.subplots(1, n_sets, figsize=(6 * n_sets, 6), sharey=True)

for i, this_set in enumerate(unique_sets):
    subset = rel_long[rel_long['set'] == this_set]
    ax = axes[i] if n_sets > 1 else axes

    sns.boxplot(data=subset, x='channel', y='relative_signal', hue='treatment',
                order=channel_order[1:], ax=ax)
    sns.stripplot(data=subset, x='channel', y='relative_signal', hue='treatment',
                  dodge=True, jitter=True, linewidth=0.5, alpha=0.6,
                  order=channel_order[1:], ax=ax)

    ax.set_title(f"Relative signal — {this_set}")
    ax.set_xlabel("Channel")
    if i == 0:
        ax.set_ylabel("Relative signal")
    else:
        ax.set_ylabel("")
    ax.legend_.remove()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()
plt.show()
