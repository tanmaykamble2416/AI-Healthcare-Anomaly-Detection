import pandas as pd

print("Starting dataset validation...")

# Load dataset
data = pd.read_csv("data/patient_vitals.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nStatistical summary:")
print(data.describe())