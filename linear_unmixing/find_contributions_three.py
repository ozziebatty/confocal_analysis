import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
import ast

print("Finding bleedthrough coefficients")

#a = 0.266
#b = 0.809

# Load characterised cells
file_path = '/Users/oskar/Desktop/BMH21_image_analysis/SBSO_colour_control/results/collective_characterised_cells.csv'
#file_path = '/Users/oskar/Desktop/steventon_lab/image_analysis/images/SBSE_code_test/results/SBSO_example_image_1_cropped/characterised_cells.csv'
df = pd.read_csv(file_path)

# Extract relevant data
X = df[["Bra", "DAPI", "z_position"]].values  # Independent variables
#X = df[["Sox1", "Bra"]].values  # Independent variables
y = df["Sox2 Cyan"].values    # Dependent variable

# Perform linear regression with no intercept
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
print(f"Estimated a: {model.coef_[0]}")
print(f"Estimated b: {model.coef_[1]}")
print(f"Estimated c: {model.coef_[2]}")
#print(f"Estimated intercept: {model.intercept_}")

df["predicted_Sox2_Cyan"] = model.predict(X)

# Compute predicted values
#df["predicted_Sox2_Orange"] = a * df["Sox1"] + b * df["Bra"]

# Compute offset (actual - predicted)
df["offset_Cyan"] = df["Sox2 Cyan"] - df["predicted_Sox2_Cyan"]

# Compute mean and standard deviation
offset_mean = df["offset_Cyan"].mean()
offset_std = df["offset_Cyan"].std()

print(f"Mean Offset: {offset_mean}")
print(f"Standard Deviation of Offset: {offset_std}")
