import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === SETTINGS ===
csv_path = r"C:\Users\ob361\data\glucose_Bra_Sox2\size.csv"
days_to_include = ['2-4', '3-4', '2-5']
sets_to_include = ['Set_1', 'Set_2', 'Set_3']

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Ensure correct dtypes
df['treatment'] = df['treatment'].astype(str)
df['replicate'] = df['replicate'].astype(str)
df['days'] = df['days'].astype(str)
df['set'] = df['set'].astype(str)

# Filter based on days and sets
df = df[df['days'].isin(days_to_include) & df['set'].isin(sets_to_include)]

# === ABSOLUTE SIGNAL — sets across, days down ===
n_rows = len(days_to_include)
n_cols = len(sets_to_include)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)

for i, day in enumerate(days_to_include):
    for j, this_set in enumerate(sets_to_include):
        ax = axes[i, j] if n_rows > 1 else axes[j]
        subset = df[(df['set'] == this_set) & (df['days'] == day)]

        sns.boxplot(data=subset, x='treatment', y='size',
                    ax=ax)
        sns.stripplot(data=subset, x='treatment', y='size',
                      dodge=True, jitter=True, linewidth=0.5, alpha=0.6,
                       ax=ax)

        ax.set_title(f"Day {day} — {this_set}")
        if j == 0:
            ax.set_ylabel("Size")
        else:
            ax.set_ylabel("")
        if i == n_rows - 1:
            ax.set_xlabel("Treatment")
        else:
            ax.set_xlabel("")
          
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left')
fig.tight_layout()
plt.show()
