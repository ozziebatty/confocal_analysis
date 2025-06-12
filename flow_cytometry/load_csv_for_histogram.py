import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
file_path = '/Users/oskar/Desktop/flow_cytometry/SBSE_d5/WT_gated.csv'
data = pd.read_csv(file_path)


# Print column names to check them
print("Columns:", data.columns.tolist())

# Strip spaces from column names (just in case)
data.columns = data.columns.str.strip()

# Now extract
fs_ca = data['FSC-A']
a638 = data['638nm660-10-A']
a488 = data['488nm525-40-A']
a561 = data['561nm610-20-A']

# Calculate the ratio
ratio = a638 / a488




# Plot the 1D histogram
plt.figure(figsize=(8, 6))
plt.hist(ratio, bins=500)
plt.xlabel('638nm660-10-A / FSC-A')
plt.ylabel('Count')
plt.title('Histogram of 638A / FSCA')

x_min = 0
x_max = 0.3
plt.xlim(x_min, x_max)


plt.tight_layout()
plt.show()
