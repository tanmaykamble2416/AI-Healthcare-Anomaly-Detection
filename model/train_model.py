import pandas as pd

print("\nStarting Feature Selection...\n")

# Load dataset
data = pd.read_csv("data/patient_vitals.csv")

# Selected features for anomaly detection
selected_features = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Derived_HRV",
    "Derived_MAP"
]

# Create feature dataset
feature_data = data[selected_features]

print("Selected Features:")
print(selected_features)

print("\nFeature Dataset Shape:")
print(feature_data.shape)

print("\nPreview of Selected Feature Data:")
print(feature_data.head())