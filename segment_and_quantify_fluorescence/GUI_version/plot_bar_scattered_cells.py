import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import pandas as pd
from scipy import stats
from itertools import combinations
import seaborn as sns

def load_gates_data(file_path):
    """Load gates data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_multiple_replicates(base_analysis_path):
    """Load gates data from multiple replicate files"""
    # Find all replicate directories
    analysis_dir = Path(base_analysis_path)
    replicate_pattern = analysis_dir / "replicate_*" / "normalised_gates" / "gates_definition.json"
    replicate_files = glob.glob(str(replicate_pattern))
    
    replicates_data = {}
    for file_path in sorted(replicate_files):
        # Extract replicate number from path
        replicate_num = Path(file_path).parent.parent.name.split('_')[-1]
        print(f"Loading replicate {replicate_num} from {file_path}")
        try:
            replicates_data[f"replicate_{replicate_num}"] = load_gates_data(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return replicates_data

def extract_channel_stats_multiple_replicates(replicates_data, gate_names):
    """Extract channel statistics for specified gates from multiple replicates"""
    all_replicates_stats = {}
    
    for replicate_name, gates_data in replicates_data.items():
        print(f"\nProcessing {replicate_name}...")
        
        # First, find all nested gates for this replicate
        all_nested_gates = {}
        for top_level_key, top_level_data in gates_data.items():
            if isinstance(top_level_data, dict):
                for nested_key, nested_data in top_level_data.items():
                    if isinstance(nested_data, dict) and 'data_analysis' in nested_data:
                        all_nested_gates[nested_key] = nested_data
        
        # Extract stats for target gates
        replicate_stats = {}
        for gate_name in gate_names:
            if gate_name in all_nested_gates:
                gate_info = all_nested_gates[gate_name]
                if 'data_analysis' in gate_info and 'channel_stats' in gate_info['data_analysis']:
                    replicate_stats[gate_name] = gate_info['data_analysis']['channel_stats']
                    print(f"  Found gate: {gate_name}")
                else:
                    print(f"  Warning: No channel_stats found for gate: {gate_name}")
            else:
                print(f"  Warning: Gate '{gate_name}' not found in {replicate_name}")
        
        all_replicates_stats[replicate_name] = replicate_stats
    
    return all_replicates_stats

def prepare_data_for_stats(all_replicates_stats, channel='channel_3'):
    """Prepare data for statistical analysis"""
    data_for_stats = []
    
    for replicate_name, replicate_stats in all_replicates_stats.items():
        for gate_name, gate_stats in replicate_stats.items():
            if channel in gate_stats:
                # Extract available fields from the gate_stats
                row_data = {
                    'replicate': replicate_name,
                    'cell_type': gate_name,
                    'opp_mean': gate_stats[channel]['mean']
                }
                
                # Add optional fields if they exist
                if 'std' in gate_stats[channel]:
                    row_data['opp_std'] = gate_stats[channel]['std']
                if 'count' in gate_stats[channel]:
                    row_data['cell_count'] = gate_stats[channel]['count']
                
                data_for_stats.append(row_data)
    
    return pd.DataFrame(data_for_stats)

def perform_statistical_tests(stats_df, test_types=['anova', 'pairwise_t'], alpha=0.05):
    """Perform statistical tests on OPP signal between cell types"""
    results = {}
    
    # Filter out 'Uncharacterised' for cleaner comparisons
    cell_types_for_comparison = ['Neural', 'Pluripotent', 'Mesoderm', 'NMP']
    comparison_df = stats_df[stats_df['cell_type'].isin(cell_types_for_comparison)]
    
    if comparison_df.empty:
        print("No data available for statistical comparison")
        return results
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF OPP SIGNAL BETWEEN CELL TYPES")
    print("="*60)
    print(f"Analyzing {len(comparison_df)} data points across {comparison_df['cell_type'].nunique()} cell types")
    print(f"Replicates per cell type:")
    for cell_type in cell_types_for_comparison:
        n_reps = len(comparison_df[comparison_df['cell_type'] == cell_type])
        if n_reps > 0:
            mean_val = comparison_df[comparison_df['cell_type'] == cell_type]['opp_mean'].mean()
            std_val = comparison_df[comparison_df['cell_type'] == cell_type]['opp_mean'].std()
            print(f"  {cell_type}: n={n_reps}, mean={mean_val:.3f}, std={std_val:.3f}")
    
    # One-way ANOVA
    if 'anova' in test_types:
        print(f"\n1. ONE-WAY ANOVA")
        print("-" * 30)
        
        # Group data by cell type
        groups = []
        group_names = []
        for cell_type in cell_types_for_comparison:
            cell_data = comparison_df[comparison_df['cell_type'] == cell_type]['opp_mean'].values
            if len(cell_data) > 0:
                groups.append(cell_data)
                group_names.append(cell_type)
                print(f"  {cell_type}: {len(cell_data)} replicates")
        
        if len(groups) >= 2 and all(len(group) > 0 for group in groups):
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
            
            print(f"\nF-statistic: {f_stat:.4f}")
            print(f"P-value: {p_value:.6f}")
            print(f"Significant at α={alpha}: {'Yes' if p_value < alpha else 'No'}")
            
            if p_value < alpha:
                print("*** ANOVA indicates significant differences between cell types ***")
            else:
                print("ANOVA indicates no significant differences between cell types")
        else:
            print("Insufficient data for ANOVA (need at least 2 groups with data)")
    
    # Pairwise t-tests (with Bonferroni correction)
    if 'pairwise_t' in test_types:
        print(f"\n2. PAIRWISE T-TESTS (Bonferroni corrected)")
        print("-" * 45)
        
        pairwise_results = {}
        # Only include cell types that have data
        available_cell_types = []
        for cell_type in cell_types_for_comparison:
            if len(comparison_df[comparison_df['cell_type'] == cell_type]) > 0:
                available_cell_types.append(cell_type)
        
        if len(available_cell_types) < 2:
            print("Insufficient cell types with data for pairwise comparisons")
            return results
            
        comparisons = list(combinations(available_cell_types, 2))
        n_comparisons = len(comparisons)
        bonferroni_alpha = alpha / n_comparisons
        
        print(f"Comparing {len(available_cell_types)} cell types: {available_cell_types}")
        print(f"Number of comparisons: {n_comparisons}")
        print(f"Bonferroni corrected α: {bonferroni_alpha:.6f}")
        print()
        
        for i, (type1, type2) in enumerate(comparisons):
            data1 = comparison_df[comparison_df['cell_type'] == type1]['opp_mean'].values
            data2 = comparison_df[comparison_df['cell_type'] == type2]['opp_mean'].values
            
            if len(data1) > 0 and len(data2) > 0:
                # Perform independent t-test
                if len(data1) == 1 and len(data2) == 1:
                    print(f"{type1} vs {type2}: Cannot perform t-test with n=1 in both groups")
                    continue
                elif len(data1) == 1 or len(data2) == 1:
                    print(f"{type1} vs {type2}: Warning - one group has n=1, t-test may be unreliable")
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # Calculate effect size (Cohen's d)
                if len(data1) > 1 and len(data2) > 1:
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                       (len(data1) + len(data2) - 2))
                else:
                    # Use combined standard deviation when n=1 in one group
                    pooled_std = np.sqrt((np.var(data1, ddof=1 if len(data1) > 1 else 0) + 
                                        np.var(data2, ddof=1 if len(data2) > 1 else 0)) / 2)
                
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                
                pairwise_results[f"{type1}_vs_{type2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant_bonferroni': p_value < bonferroni_alpha,
                    'significant_uncorrected': p_value < alpha,
                    'n1': len(data1),
                    'n2': len(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2)
                }
                
                print(f"{type1} vs {type2} (n₁={len(data1)}, n₂={len(data2)}):")
                print(f"  Means:       {np.mean(data1):8.4f} vs {np.mean(data2):8.4f}")
                print(f"  t-statistic: {t_stat:8.4f}")
                print(f"  p-value:     {p_value:8.6f}")
                print(f"  Cohen's d:   {cohens_d:8.4f}")
                print(f"  Significant (Bonferroni): {'Yes' if p_value < bonferroni_alpha else 'No'}")
                print(f"  Significant (uncorrected): {'Yes' if p_value < alpha else 'No'}")
                
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    effect_size = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_size = "small"
                elif abs(cohens_d) < 0.8:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                print(f"  Effect size: {effect_size}")
                print()
        
        results['pairwise_t'] = pairwise_results
    
    return results

def plot_opp_channel_with_stats(all_replicates_stats, statistical_results=None):
    """
    Create a focused plot for OPP channel with individual points, error bars, 
    mean values, and statistical significance annotations
    """
    # Extract OPP data (channel_3) from all replicates
    opp_data = {}
    
    # Get all unique gate names
    all_gate_names = set()
    for replicate_stats in all_replicates_stats.values():
        all_gate_names.update(replicate_stats.keys())
    
    # Extract OPP values for each gate across replicates
    for gate_name in sorted(all_gate_names):
        opp_values = []
        replicate_names = []
        
        for replicate_name, replicate_stats in all_replicates_stats.items():
            if gate_name in replicate_stats and 'channel_3' in replicate_stats[gate_name]:
                opp_values.append(replicate_stats[gate_name]['channel_3']['mean'])
                replicate_names.append(replicate_name)
        
        if opp_values:
            opp_data[gate_name] = {
                'values': opp_values,
                'replicates': replicate_names,
                'mean': np.mean(opp_values),
                'sem': np.std(opp_values) / np.sqrt(len(opp_values)) if len(opp_values) > 1 else 0,
                'std': np.std(opp_values) if len(opp_values) > 1 else 0,
                'n': len(opp_values)
            }
    
    if not opp_data:
        print("No OPP data found to plot")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors for different cell types
    gate_colors = {
        'Uncharacterised': '#808080',  # Gray
        'Neural': '#2E8B57',           # Green (Sea Green)
        'Pluripotent': '#87CEEB',      # Pale Blue (Sky Blue)
        'Mesoderm': '#DC143C',         # Red (Crimson)
        'NMP': '#FF8C00'               # Orange (Dark Orange)
    }
    
    # All points will be circles - define unique replicates for legend
    unique_replicates = sorted(list(set([rep for data in opp_data.values() for rep in data['replicates']])))
    
    gate_names = list(opp_data.keys())
    x_positions = np.arange(len(gate_names))
    
    # Plot bars with error bars
    bars = []
    for i, gate_name in enumerate(gate_names):
        data = opp_data[gate_name]
        color = gate_colors.get(gate_name, '#999999')
        
        # Plot bar with SEM error bars
        bar = ax.bar(i, data['mean'], 
                    color=color, alpha=0.6, 
                    yerr=data['sem'], capsize=5, 
                    edgecolor='black', linewidth=1.2)
        bars.append(bar)
        
        # Add mean value text on top of bar
        ax.text(i, data['mean'] + data['sem'] + (ax.get_ylim()[1] * 0.02), 
               f'{data["mean"]:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
        

    
    # Plot individual data points with different markers for replicates
    for i, gate_name in enumerate(gate_names):
        data = opp_data[gate_name]
        
        # Add some jitter to x-position
        x_jitter = np.random.normal(0, 0.1, len(data['values']))
        x_positions_jittered = [i + jit for jit in x_jitter]
        
        # Plot each point as a black circle
        for j, (x_pos, y_val, rep_name) in enumerate(zip(x_positions_jittered, data['values'], data['replicates'])):
            ax.scatter(x_pos, y_val, 
                      marker='o', s=80, 
                      color='black', 
                      zorder=10, alpha=0.9)
    

    
    # Customize the plot
    ax.set_xlabel('Cell Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('OPP Mean Expression', fontsize=14, fontweight='bold')
    ax.set_title('OPP Expression by Cell Type\n(Individual points, means ± SEM)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(range(len(gate_names)))
    ax.set_xticklabels(gate_names, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    

    
    # No need to adjust y-axis for significance brackets since they're removed
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("OPP EXPRESSION SUMMARY STATISTICS")
    print("="*60)
    
    for gate_name in gate_names:
        data = opp_data[gate_name]
        print(f"\n{gate_name}:")
        print(f"  Mean ± SEM: {data['mean']:.3f} ± {data['sem']:.3f}")
        print(f"  Mean ± SD:  {data['mean']:.3f} ± {data['std']:.3f}")
        print(f"  n = {data['n']} replicates")
        print(f"  Individual values: {[f'{v:.3f}' for v in data['values']]}")

def main():
    # Path to the main SBSO_analysis folder
    analysis_folder = "/Users/oskar/Desktop/SBSO_analysis"
    
    # Gates to analyze
    target_gates = ['Uncharacterised', 'Neural', 'Pluripotent', 'Mesoderm', 'NMP']
    
    try:
        # Load data from multiple replicates
        print("Loading gates data from multiple replicates...")
        replicates_data = load_multiple_replicates(analysis_folder)
        
        if not replicates_data:
            print("No replicate files found!")
            print(f"Looking for files matching: {analysis_folder}/replicate_*/normalised_gates/gates_definition.json")
            return
            
        print(f"Loaded {len(replicates_data)} replicates")
        
        # Extract channel stats for target gates across all replicates
        print("\nExtracting channel statistics across replicates...")
        all_replicates_stats = extract_channel_stats_multiple_replicates(replicates_data, target_gates)
        
        if any(all_replicates_stats.values()):
            # Prepare data for statistical analysis
            print("\nPreparing data for statistical analysis...")
            stats_df = prepare_data_for_stats(all_replicates_stats, channel='channel_3')
            
            statistical_results = None
            if not stats_df.empty:
                # Perform statistical tests
                statistical_results = perform_statistical_tests(
                    stats_df, 
                    test_types=['anova', 'pairwise_t'],
                    alpha=0.05
                )
            
            # Create the enhanced OPP plot
            print("\nCreating enhanced OPP channel plot...")
            plot_opp_channel_with_stats(all_replicates_stats, statistical_results)
            
            # Export results to CSV for further analysis
            if not stats_df.empty:
                stats_df.to_csv('opp_statistical_data.csv', index=False)
                print("\nStatistical data exported to 'opp_statistical_data.csv'")
            
        else:
            print("No valid gate statistics found!")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()