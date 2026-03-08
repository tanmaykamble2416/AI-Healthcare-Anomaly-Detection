import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

print("\nAI Healthcare Anomaly Detection Pipeline\n")

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
data = pd.read_csv("data/patient_vitals.csv")

# ------------------------------------------------
# 2. Feature Selection
# ------------------------------------------------
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

print("Feature Preview:")
print(df_vitals.head())

# ------------------------------------------------
# 3. Feature Scaling (Activity 2.3)
# ------------------------------------------------
scaler = MinMaxScaler()

scaled_vitals = scaler.fit_transform(df_vitals)

df_scaled = pd.DataFrame(
    scaled_vitals,
    columns=df_vitals.columns
)

print("\nScaled Feature Preview:")
print(df_scaled.head())

# ------------------------------------------------
# 4. Temporal Modeling (Activity 3.1)
# ------------------------------------------------
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

# ------------------------------------------------
# 5. Prepare Data for Model Training
# ------------------------------------------------
X_train = sequences.reshape(sequences.shape[0], -1)

# ------------------------------------------------
# 6. Autoencoder Model Training (Activity 3.2)
# ------------------------------------------------
print("\nActivity 3.2: Autoencoder Model Training\n")

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)

# Decoder
decoder = Dense(32, activation="relu")(encoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

# Build model
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("Training Autoencoder...")

autoencoder.fit(
    X_train,
    X_train,
    epochs=10,
    batch_size=64,
    shuffle=True,
    verbose=1
)

print("\nModel training completed.")

# ------------------------------------------------
# 7. Save Model
# ------------------------------------------------
autoencoder.save("model/autoencoder_model.h5")

print("Model saved successfully.")


# ------------------------------------------------
# 8. Activity 3.3 – Anomaly Scoring & Classification
# ------------------------------------------------
print("\nActivity 3.3: Anomaly Scoring & Severity Classification\n")

# Reconstruct the training data
reconstructed = autoencoder.predict(X_train)

# Calculate reconstruction error (Mean Squared Error)
reconstruction_error = np.mean((X_train - reconstructed) ** 2, axis=1)

# Convert to DataFrame for analysis
error_df = pd.DataFrame({
    "reconstruction_error": reconstruction_error
})

# Determine anomaly threshold (for example: 95th percentile)
threshold = np.percentile(reconstruction_error, 95)

print("Anomaly detection threshold:", threshold)

# Label anomalies
error_df["is_anomaly"] = error_df["reconstruction_error"] > threshold

# Severity classification
def classify_severity(err):
    if err > threshold * 2:
        return "High"
    elif err > threshold * 1.5:
        return "Medium"
    elif err > threshold:
        return "Low"
    else:
        return "Normal"

error_df["severity"] = error_df["reconstruction_error"].apply(classify_severity)

# Preview results
print("\nSample anomaly scoring results:")
print(error_df.head(10))

# Count anomalies
print("\nAnomaly summary:")
print(error_df["severity"].value_counts())