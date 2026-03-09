from kafka import KafkaProducer
import pandas as pd
import json
import time

print("Starting Kafka Producer...")

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

data = pd.read_csv("data/patient_vitals.csv")

for _, row in data.iterrows():
    message = row.to_dict()

    producer.send("patient_vitals", value=message)

    print("Sent:", message)

    time.sleep(1)