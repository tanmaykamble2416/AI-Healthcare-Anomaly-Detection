import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("\nAI Healthcare Anomaly Detection Pipeline\n")

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("data/patient_vitals.csv")

# ---------------------------
# Feature Selection
# ---------------------------
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

df_vitals = data[selected_features]

print("Feature preview:")
print(df_vitals.head())

# ---------------------------
# Feature Scaling (Activity 2.3)
# ---------------------------
scaler = MinMaxScaler()

scaled_vitals = scaler.fit_transform(df_vitals)

df_scaled = pd.DataFrame(
    scaled_vitals,
    columns=df_vitals.columns
)

print("\nScaled Feature Preview:")
print(df_scaled.head())

# ---------------------------
# Activity 3.1 Temporal Modeling
# ---------------------------
print("\nActivity 3.1: Temporal Modeling (Sliding Window)\n")

window_size = 10

data_array = df_scaled.values

sequences = []

for i in range(len(data_array) - window_size):
    window = data_array[i:i + window_size]
    sequences.append(window)

sequences = np.array(sequences)

print("Temporal sequences created successfully.")
print("Sequence shape:", sequences.shape)