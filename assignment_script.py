

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset


# Load the dataset
url = 'https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv'

# Load the dataset directly from the URL
df = pd.read_csv(url)


# Check the column names to ensure they match the expected ones
print(df.columns)

# Descriptive statistics for the retention rates
control_group = df[df['version'] == 'gate_30']
treatment_group = df[df['version'] == 'gate_40']

# Convert boolean values to integers (1 for True, 0 for False) for 1-day retention
control_1d = control_group['retention_1'].astype(int)
treatment_1d = treatment_group['retention_1'].astype(int)

# Convert boolean values to integers (1 for True, 0 for False) for 7-day retention
control_7d = control_group['retention_7'].astype(int)
treatment_7d = treatment_group['retention_7'].astype(int)

# Plotting histograms of 1-day retention for both groups using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(control_1d, bins=30, alpha=0.5, label="Control Group (Gate 30)", color='blue')
plt.hist(treatment_1d, bins=30, alpha=0.5, label="Treatment Group (Gate 40)", color='orange')
plt.title("1-Day Retention Comparison")
plt.xlabel("Retention (1-Day)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plotting histograms of 7-day retention for both groups using matplotlib
plt.figure(figsize=(10, 6))
plt.hist(control_7d, bins=30, alpha=0.5, label="Control Group (Gate 30)", color='blue')
plt.hist(treatment_7d, bins=30, alpha=0.5, label="Treatment Group (Gate 40)", color='orange')
plt.title("7-Day Retention Comparison")
plt.xlabel("Retention (7-Day)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
