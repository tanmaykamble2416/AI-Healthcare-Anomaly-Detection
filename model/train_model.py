import pandas as pd

print("\nHealthcare Dataset Validation Started\n")

# Load dataset
data = pd.read_csv("data/patient_vitals.csv")

# 1. Dataset preview
print("Dataset Preview (first 5 rows):\n")
print(data.head().to_string(index=False))

# 2. Dataset shape and size
print("\nDataset Shape (Rows, Columns):")
print(data.shape)

# 3. Check feature availability
print("\nAvailable Features:")
print(list(data.columns))

# 4. Missing values check
print("\nMissing Values Check:")
print(data.isnull().sum())

# 5. Numerical range validation
print("\nStatistical Summary (Range Consistency Check):")
print(data.describe())

print("\nDataset validation completed successfully.")
