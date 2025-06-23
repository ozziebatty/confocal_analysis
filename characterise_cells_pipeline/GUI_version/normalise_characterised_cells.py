import pandas as pd
import numpy as np
import os
from pathlib import Path

def normalize_characterised_cells(replicate_number):
    """
    Normalize characterised cells data so that the average value across all cells 
    for each channel is 1.
    """
    replicate_string = f"replicate_{replicate_number}"
    print(f"Loading data for replicate_{replicate_number}")
    
    # Set up paths
    project_path = os.path.normpath('/Users/oskar/Desktop/SBSO_analysis')
    replicate_path = os.path.join(project_path, replicate_string)
    characterised_cells_path = os.path.join(replicate_path, f"{replicate_string}_characterised_cells.csv")
    
    try:
        # Load the data
        print(f"  Loading: {characterised_cells_path}")
        df = pd.read_csv(characterised_cells_path)
        
        # Print original data info
        print(f"  Loaded {len(df)} cells")
        print(f"  Columns: {list(df.columns)}")
        
        # Identify channel columns (assuming channels 0-4)
        channel_columns = ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4']
        
        # Check which channel columns exist
        existing_channels = [col for col in channel_columns if col in df.columns]
        print(f"  Found channel columns: {existing_channels}")
        
        # Create a copy for normalization
        df_normalized = df.copy()
        
        # Normalize each channel so mean = 1
        normalization_factors = {}
        for channel in existing_channels:
            # Calculate mean across all cells for this channel
            channel_mean = df[channel].mean()
            normalization_factors[channel] = channel_mean
            
            # Normalize: divide by mean so new mean = 1
            df_normalized[channel] = df[channel] / channel_mean
            
            print(f"  {channel}: original mean = {channel_mean:.3f}, normalized mean = {df_normalized[channel].mean():.3f}")
        
        # Save normalized data
        output_path = os.path.join(replicate_path, f"normalised_characterised_cells_{replicate_string}.csv")
        df_normalized.to_csv(output_path, index=False)
        print(f"  Saved normalized data: {output_path}")
        
        # Save normalization factors for reference
        factors_path = os.path.join(replicate_path, f"normalization_factors_{replicate_string}.csv")
        factors_df = pd.DataFrame(list(normalization_factors.items()), 
                                columns=['channel', 'original_mean'])
        factors_df.to_csv(factors_path, index=False)
        print(f"  Saved normalization factors: {factors_path}")
        
        return df_normalized, normalization_factors
        
    except FileNotFoundError:
        print(f"  ERROR: File not found: {characterised_cells_path}")
        return None, None
    except Exception as e:
        print(f"  ERROR processing replicate {replicate_number}: {e}")
        return None, None

def normalize_all_replicates():
    """
    Normalize characterised cells data for all available replicates.
    """
    print("Normalizing characterised cells data for all replicates...")
    print("="*60)
    
    project_path = '/Users/oskar/Desktop/SBSO_analysis'
    
    # Find all replicate directories
    replicate_dirs = []
    for item in os.listdir(project_path):
        if item.startswith('replicate_') and os.path.isdir(os.path.join(project_path, item)):
            try:
                replicate_num = int(item.split('_')[1])
                replicate_dirs.append(replicate_num)
            except (IndexError, ValueError):
                continue
    
    replicate_dirs.sort()
    print(f"Found replicates: {replicate_dirs}")
    print()
    
    successful_normalizations = []
    all_normalization_factors = {}
    
    # Process each replicate
    for replicate_num in replicate_dirs:
        df_norm, factors = normalize_characterised_cells(replicate_num)
        if df_norm is not None:
            successful_normalizations.append(replicate_num)
            all_normalization_factors[f"replicate_{replicate_num}"] = factors
        print()
    
    # Summary
    print("="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Successfully normalized {len(successful_normalizations)} replicates: {successful_normalizations}")
    
    if all_normalization_factors:
        print("\nNormalization factors (original means) for each replicate:")
        for replicate, factors in all_normalization_factors.items():
            print(f"\n{replicate}:")
            for channel, factor in factors.items():
                print(f"  {channel}: {factor:.3f}")
    
    # Save combined normalization factors
    if all_normalization_factors:
        combined_factors = []
        for replicate, factors in all_normalization_factors.items():
            for channel, factor in factors.items():
                combined_factors.append({
                    'replicate': replicate,
                    'channel': channel,
                    'original_mean': factor
                })
        
        combined_df = pd.DataFrame(combined_factors)
        output_path = os.path.join(project_path, 'all_normalization_factors.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nSaved combined normalization factors: {output_path}")

if __name__ == "__main__":
    # Run normalization for all replicates
    normalize_all_replicates()
    
    # Example: normalize just one replicate
    # replicate_number = 4
    # df_normalized, factors = normalize_characterised_cells(replicate_number)