from kafka import KafkaConsumer
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print("Loading trained model...")

model = load_model("model/autoencoder_model.h5", compile=False)

print("Model loaded successfully")
print("Model input shape:", model.input_shape)

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

scaler = MinMaxScaler()

consumer = KafkaConsumer(
    "patient_vitals",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Kafka Consumer started... Monitoring patients\n")

window_size = 10
buffer = []

for message in consumer:

    data = message.value

    features = [data[f] for f in selected_features]

    buffer.append(features)

    if len(buffer) > window_size:
        buffer.pop(0)

    if len(buffer) < window_size:
        continue

    df = pd.DataFrame(buffer, columns=selected_features)

    scaled = scaler.fit_transform(df)

    X = scaled.reshape(1, -1)

    reconstruction = model.predict(X)

    error = np.mean((X - reconstruction) ** 2)

    threshold = 0.075

    if error > threshold:
        status = "ANOMALY DETECTED"
    else:
        status = "Normal"

    print("\nPatient Window Ending ID:", data["Patient ID"])
    print("Anomaly Score:", error)
    print("Status:", status)