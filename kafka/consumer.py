from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "patient_vitals",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="healthcare-monitor",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print("Kafka Consumer started...")

for message in consumer:
    print("Received:", message.value)