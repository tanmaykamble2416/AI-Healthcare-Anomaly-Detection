import pandas as pd

print("\n==============================")
print("Patient Vitals Dataset Loaded")
print("==============================\n")

# Load dataset
data = pd.read_csv("data/patient_vitals.csv")

# Display first 10 rows in table format
print("Preview of Dataset:\n")
print(data.head(10).to_string(index=False))

print("\n----------------------------------")
print("Dataset Shape (Rows, Columns):")
print("----------------------------------")
print(data.shape)

print("\n----------------------------------")
print("Column Names:")
print("----------------------------------")
print(list(data.columns))

print("\n----------------------------------")
print("Missing Values Check:")
print("----------------------------------")
print(data.isnull().sum())

print("\n----------------------------------")
print("Statistical Summary:")
print("----------------------------------")
print(data.describe())