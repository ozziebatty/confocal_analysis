#%% IMPORT PACKAGES

print("Cell Type Analysis Script")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import to_rgba

#%% LOAD DATA

# ========== Configure settings ==========
# Base folder containing the consolidated cell data
base_folder = os.path.normpath(r'Y:\Room225_SharedFolder\CytoflexLX_data\oskar\relevant_fcs_files\d2_treatments\B')

# Input file from previous script
consolidated_csv_path = os.path.join(base_folder, 'consolidated_cell_data.csv')

# Output folder for cell type analysis
output_folder = base_folder

# Check if consolidated data exists
if not os.path.exists(consolidated_csv_path):
    raise FileNotFoundError(f"Consolidated cell data not found at: {consolidated_csv_path}")
    
print(f"Loading consolidated cell data from: {consolidated_csv_path}")
df = pd.read_csv(consolidated_csv_path)

print(f"Loaded {len(df)} cells across {df['Treatment'].nunique()} treatments")
print(f"Treatments: {df['Treatment'].unique()}")

#%% DEFINE CELL TYPES

def classify_cell_type(row):
    """
    Classify cell type based on fluorescence combinations
    
    Cell type definitions:
    - Pluripotent = 488-, 561-, 638+
    - Neural = 488+, 561-, 638+ OR 488+, 561-, 638-
    - Mesoderm = 488-, 561+, 638-
    - NMP = 488+, 561+, 638- OR 488+, 561+, 638+ OR 488-, 561+, 638+
    - Undefined = 488-, 561-, 638-
    """
    
    pos_488 = row['488nm525-40-A_pos']
    pos_561 = row['561nm610-20-A_pos'] 
    pos_638 = row['638nm660-10-A_pos']
    
    # Pluripotent: 488-, 561-, 638+
    if not pos_488 and not pos_561 and pos_638:
        return 'Pluripotent'
    
    # Neural: 488+, 561-, 638+ OR 488+, 561-, 638-
    elif pos_488 and not pos_561 and (pos_638 or not pos_638):
        return 'Neural'
    
    # Mesoderm: 488-, 561+, 638-
    elif not pos_488 and pos_561 and not pos_638:
        return 'Mesoderm'
    
    # NMP: 488+, 561+, 638- OR 488+, 561+, 638+ OR 488-, 561+, 638+
    elif (pos_488 and pos_561 and not pos_638) or \
         (pos_488 and pos_561 and pos_638) or \
         (not pos_488 and pos_561 and pos_638):
        return 'NMP'
    
    # Undefined: 488-, 561-, 638- (and any other combinations not defined above)
    else:
        return 'Undefined'

# Apply cell type classification
print("\n===== CLASSIFYING CELL TYPES =====")
df['Cell_Type'] = df.apply(classify_cell_type, axis=1)

# Print classification summary
print("\nCell type classification summary:")
cell_type_counts = df['Cell_Type'].value_counts()
print(cell_type_counts)
print(f"\nTotal classified cells: {len(df)}")

#%% ANALYZE CELL TYPE PROPORTIONS

# Calculate proportions for each treatment
treatment_cell_types = df.groupby(['Treatment', 'Cell_Type']).size().unstack(fill_value=0)

# Calculate percentages
treatment_percentages = treatment_cell_types.div(treatment_cell_types.sum(axis=1), axis=0) * 100

# Display results
print("\n===== CELL TYPE PROPORTIONS BY TREATMENT =====")
print("\nAbsolute counts:")
print(treatment_cell_types)
print("\nPercentages:")
print(treatment_percentages.round(2))

# Save detailed results
detailed_results_path = os.path.join(output_folder, 'cell_type_analysis_detailed.csv')
treatment_cell_types.to_csv(detailed_results_path)
print(f"\nDetailed cell type counts saved to: {detailed_results_path}")

percentage_results_path = os.path.join(output_folder, 'cell_type_percentages.csv')
treatment_percentages.to_csv(percentage_results_path)
print(f"Cell type percentages saved to: {percentage_results_path}")

#%% CREATE PIE CHARTS

# Define colors for each cell type
cell_type_colors = {
    'Neural': '#2E8B57',        # Green
    'Pluripotent': '#87CEEB',   # Light blue  
    'Mesoderm': '#DC143C',      # Red
    'NMP': '#FF8C00',           # Orange
    'Undefined': '#808080'      # Grey
}

# Get all treatments
treatments = df['Treatment'].unique()
n_treatments = len(treatments)

# Calculate subplot layout
cols = min(3, n_treatments)  # Max 3 columns
rows = (n_treatments + cols - 1) // cols  # Ceiling division

# Create figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

# Handle single treatment case
if n_treatments == 1:
    axes = [axes]
elif rows == 1:
    axes = axes if hasattr(axes, '__iter__') else [axes]
else:
    axes = axes.flatten()

# Create pie chart for each treatment
for i, treatment in enumerate(treatments):
    ax = axes[i]
    
    # Get data for this treatment
    treatment_data = treatment_percentages.loc[treatment]
    
    # Filter out cell types with 0 counts
    treatment_data = treatment_data[treatment_data > 0]
    
    # Get colors for the cell types present
    colors = [cell_type_colors.get(cell_type, '#000000') for cell_type in treatment_data.index]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(treatment_data.values, 
                                     labels=treatment_data.index,
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     textprops={'fontsize': 10})
    
    # Customize the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f'{treatment}\n(n = {treatment_cell_types.loc[treatment].sum():,} cells)', 
                fontsize=12, fontweight='bold')

# Hide unused subplots
for i in range(n_treatments, len(axes)):
    axes[i].axis('off')

# Add overall title and legend
plt.suptitle('Cell Type Distributions by Treatment', fontsize=16, fontweight='bold', y=0.98)

# Create legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=cell_type) 
                  for cell_type, color in cell_type_colors.items()]
fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
          ncol=len(cell_type_colors), fontsize=11)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12)

# Save the figure
pie_chart_path = os.path.join(output_folder, 'cell_type_pie_charts.png')
plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nPie charts saved to: {pie_chart_path}")

#%% CREATE SUMMARY BAR CHART

# Create a summary bar chart showing all treatments side by side
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for stacked bar chart
cell_types = list(cell_type_colors.keys())
treatments_list = list(treatments)

# Create matrix for plotting
plot_data = np.zeros((len(treatments_list), len(cell_types)))

for i, treatment in enumerate(treatments_list):
    for j, cell_type in enumerate(cell_types):
        if cell_type in treatment_percentages.columns:
            plot_data[i, j] = treatment_percentages.loc[treatment, cell_type]

# Create stacked bar chart
bottom = np.zeros(len(treatments_list))
bars = []

for j, cell_type in enumerate(cell_types):
    bar = ax.bar(treatments_list, plot_data[:, j], bottom=bottom, 
                color=cell_type_colors[cell_type], label=cell_type, 
                edgecolor='white', linewidth=0.5)
    bars.append(bar)
    bottom += plot_data[:, j]

# Customize the plot
ax.set_ylabel('Percentage of Cells', fontsize=12, fontweight='bold')
ax.set_xlabel('Treatment', fontsize=12, fontweight='bold')
ax.set_title('Cell Type Distribution Across Treatments', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
ax.set_ylim(0, 100)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Rotate x-axis labels if needed
if len(max(treatments_list, key=len)) > 8:
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# Save the bar chart
bar_chart_path = os.path.join(output_folder, 'cell_type_bar_chart.png')
plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Bar chart saved to: {bar_chart_path}")

#%% CREATE RELATIVE TO UNTREATED PLOTS

print("\n===== CREATING PLOTS RELATIVE TO UNTREATED =====")

# Check if untreated control exists
if 'untreated' in treatments:
    print("Found 'untreated' control. Creating relative plots...")
    
    # Get untreated percentages as baseline
    untreated_percentages = treatment_percentages.loc['untreated']
    
    # Calculate fold changes relative to untreated
    relative_data = treatment_percentages.copy()
    
    # Calculate fold change for each treatment
    for treatment in treatments:
        if treatment != 'untreated':
            for cell_type in cell_types:
                untreated_value = untreated_percentages.get(cell_type, 0.1)  # Use 0.1 as minimum to avoid division by zero
                current_value = treatment_percentages.loc[treatment, cell_type]
                
                # Calculate fold change (avoid division by zero)
                if untreated_value > 0:
                    relative_data.loc[treatment, cell_type] = current_value / untreated_value
                else:
                    relative_data.loc[treatment, cell_type] = current_value / 0.1  # Arbitrary small number
    
    # Set untreated to 1.0 (baseline)
    relative_data.loc['untreated'] = 1.0
    
    # Save fold change data
    fold_change_path = os.path.join(output_folder, 'cell_type_fold_changes.csv')
    relative_data.to_csv(fold_change_path)
    print(f"Fold change data saved to: {fold_change_path}")
    
    # Create bar chart showing fold changes
    treated_only = [t for t in treatments if t != 'untreated']
    
    if len(treated_only) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        x = np.arange(len(treated_only))
        width = 0.15  # Width of bars
        
        # Create bars for each cell type
        for i, cell_type in enumerate(cell_types):
            values = [relative_data.loc[treatment, cell_type] for treatment in treated_only]
            bars = ax.bar(x + i * width, values, width, 
                         label=cell_type, color=cell_type_colors[cell_type], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{value:.2f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        # Add horizontal line at y=1 (no change)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(len(treated_only)/2, 1.05, 'No change (untreated level)', 
               ha='center', va='bottom', fontsize=10, style='italic')
        
        # Customize plot
        ax.set_xlabel('Treatment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fold Change vs Untreated', fontsize=12, fontweight='bold')
        ax.set_title('Cell Type Changes Relative to Untreated Control', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)  # Center the group labels
        ax.set_xticklabels(treated_only)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(relative_data.max()) * 1.2)
        
        plt.tight_layout()
        
        # Save fold change bar chart
        fold_bar_path = os.path.join(output_folder, 'cell_type_fold_change_bars.png')
        plt.savefig(fold_bar_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fold change bar chart saved to: {fold_bar_path}")
    
    # Create difference plot (percentage point changes)
    difference_data = treatment_percentages.copy()
    
    for treatment in treatments:
        if treatment != 'untreated':
            for cell_type in cell_types:
                untreated_pct = untreated_percentages.get(cell_type, 0)
                current_pct = treatment_percentages.loc[treatment, cell_type]
                difference_data.loc[treatment, cell_type] = current_pct - untreated_pct
    
    # Set untreated to 0 (no difference from itself)
    difference_data.loc['untreated'] = 0.0
    
    # Save difference data
    difference_path = os.path.join(output_folder, 'cell_type_differences.csv')
    difference_data.to_csv(difference_path)
    print(f"Difference data saved to: {difference_path}")
    
    # Print summary of changes
    print("\n===== SUMMARY OF CHANGES RELATIVE TO UNTREATED =====")
    for treatment in treated_only:
        print(f"\n{treatment} vs Untreated:")
        for cell_type in cell_types:
            fold_change = relative_data.loc[treatment, cell_type]
            diff = difference_data.loc[treatment, cell_type]
            print(f"  {cell_type}: {fold_change:.2f}x fold change ({diff:+.1f} percentage points)")
    
else:
    print("No 'untreated' control found. Skipping relative analysis.")
    print("Available treatments:", list(treatments))

#%% SAVE INDIVIDUAL CELL TYPE DATA

print("\n===== SAVING INDIVIDUAL CELL TYPE DATA =====")

# Save individual CSV files for each cell type across all treatments
for cell_type in cell_types:
    if cell_type in df['Cell_Type'].values:
        cell_type_data = df[df['Cell_Type'] == cell_type].copy()
        cell_type_path = os.path.join(output_folder, f'{cell_type.lower()}_cells.csv')
        cell_type_data.to_csv(cell_type_path, index=False)
        print(f"{cell_type} cells saved to: {cell_type_path} ({len(cell_type_data)} cells)")

#%% SUMMARY STATISTICS

print("\n===== SUMMARY STATISTICS =====")

# Overall statistics
total_cells = len(df)
print(f"Total cells analyzed: {total_cells:,}")

# Treatment statistics  
for treatment in treatments:
    treatment_df = df[df['Treatment'] == treatment]
    print(f"\n{treatment}:")
    print(f"  Total cells: {len(treatment_df):,}")
    
    treatment_counts = treatment_df['Cell_Type'].value_counts()
    treatment_pcts = (treatment_counts / len(treatment_df) * 100).round(1)
    
    for cell_type in cell_types:
        count = treatment_counts.get(cell_type, 0)
        pct = treatment_pcts.get(cell_type, 0.0)
        print(f"  {cell_type}: {count:,} ({pct}%)")

# Create summary table
summary_data = []
for treatment in treatments:
    treatment_df = df[df['Treatment'] == treatment]
    row = {'Treatment': treatment, 'Total_Cells': len(treatment_df)}
    
    treatment_counts = treatment_df['Cell_Type'].value_counts()
    treatment_pcts = (treatment_counts / len(treatment_df) * 100).round(1)
    
    for cell_type in cell_types:
        row[f'{cell_type}_Count'] = treatment_counts.get(cell_type, 0)
        row[f'{cell_type}_Percent'] = treatment_pcts.get(cell_type, 0.0)
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_folder, 'cell_type_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary statistics saved to: {summary_path}")

print("\n===== ANALYSIS COMPLETE =====")
print("Cell type analysis completed successfully!")
print("Check the output folder for:")
print("- Pie charts showing cell type distributions")
print("- Bar chart comparing treatments")
print("- CSV files with detailed counts and percentages")
print("- Individual cell type data files")