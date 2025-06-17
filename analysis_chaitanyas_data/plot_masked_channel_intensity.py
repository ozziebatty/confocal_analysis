import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === SETTINGS ===
csv_path = r"C:\Users\ob361\data\glucose_Bra_Sox2\masked_channel_intensity_means.csv"
days_to_include = ['2-4', '3-4', '2-5']
sets_to_include = ['Set_1', 'Set_2', 'Set_3']

show_dapi = False  # Toggle this to False to hide DAPI

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Ensure correct dtypes
df['treatment'] = df['treatment'].astype(str)
df['replicate'] = df['replicate'].astype(str)
df['days'] = df['days'].astype(str)
df['set'] = df['set'].astype(str)
df['inverse'] = df['inverse'].astype(str)

# Filter based on days and sets
#df = df[df['days'].isin(days_to_include) & df['set'].isin(sets_to_include)]
print(df)

# Filter based on days, sets and inverse
df = df[df['days'].isin(days_to_include) & df['set'].isin(sets_to_include) & df['inverse'].isin(['False'])]

print(df)

# Melt for absolute signal
signal_cols = [f'channel_{i}' for i in range(4)]
df_long = df.melt(
    id_vars=['image_name', 'inverse', 'treatment', 'set', 'replicate', 'days'],
    value_vars=signal_cols,
    var_name='channel',
    value_name='signal'
)

# Optionally remove DAPI (channel_0)
if not show_dapi:
    df_long = df_long[df_long['channel'] != 'channel_0']

# Colour palette and channel order
channel_order = ['channel_0', 'channel_1', 'channel_2', 'channel_3']
plot_order = [ch for ch in channel_order if show_dapi or ch != 'channel_0']

# === ABSOLUTE SIGNAL — sets across, days down ===
n_rows = len(days_to_include)
n_cols = len(sets_to_include)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)

for i, day in enumerate(days_to_include):
    for j, this_set in enumerate(sets_to_include):
        ax = axes[i, j] if n_rows > 1 else axes[j]
        subset = df_long[(df_long['set'] == this_set) & (df_long['days'] == day)]

        sns.boxplot(data=subset, x='channel', y='signal', hue ='treatment',
                    order=plot_order, ax=ax)
        sns.stripplot(data=subset, x='channel', y='signal', hue='treatment',
                      dodge=True, jitter=True, linewidth=0.5, alpha=0.6,
                      order=plot_order, ax=ax)

        ax.set_title(f"Day {day} — {this_set}")
        if j == 0:
            ax.set_ylabel("Signal")
        else:
            ax.set_ylabel("")
        if i == n_rows - 1:
            ax.set_xlabel("Channel")
        else:
            ax.set_xlabel("")

     #   if not day == '2-5':
            #ax.legend_.remove()
    #    else:
           # if not this_set == 'Set_3':
                #ax.legend_.remove()
          
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()
plt.show()
