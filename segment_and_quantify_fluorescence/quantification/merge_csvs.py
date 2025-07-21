import pandas as pd

print("Running...")

input_file_pathways = [
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p1/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p2/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p3/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p4/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p5/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p6/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p7/characterised_cells.csv',
    '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/samples_1_to_8_2024_12_23__21_34_35__p8/characterised_cells.csv',
]

# Output file path
output_file = '/Users/oskar/Desktop/BMH21_image_analysis/SBSE_control/results/concatenated_control_characterised_cells.csv'

# Read the first file as the base DataFrame
concatenated_file = pd.read_csv(input_file_pathways[0])

# Append the rest, ensuring they match the columns of the first file
for file_pathway in input_file_pathways[1:]:
    file_to_concatenate = pd.read_csv(file_pathway)
    concatenated_file = pd.concat([concatenated_file, file_to_concatenate], ignore_index=True)

# Save the merged DataFrame
concatenated_file.to_csv(output_file, index=False)

print(f'Merged file saved to {output_file}')
