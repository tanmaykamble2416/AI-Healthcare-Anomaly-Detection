import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("\nActivity 2.3: Feature Scaling\n")

# Load dataset
data = pd.read_csv("data/patient_vitals.csv")

# Selected features
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

# Extract vital features
df_vitals = data[selected_features]

# Initialize scaler
scaler = MinMaxScaler()

# Apply scaling
scaled_vitals = scaler.fit_transform(df_vitals)

# Convert back to dataframe
df_scaled = pd.DataFrame(
    scaled_vitals,
    columns=df_vitals.columns
)

print("Scaled Feature Preview:")
print(df_scaled.head())

print("\nScaled Feature Info:")
print(df_scaled.info())