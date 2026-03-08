import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Exploratory Data Analysis")

data = pd.read_csv("data/patient_vitals.csv")

vital_features = [
    "Heart Rate",
    "Oxygen Saturation",
    "Body Temperature",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure"
]

plt.figure(figsize=(12,8))

for i, feature in enumerate(vital_features, 1):
    plt.subplot(2,3,i)
    sns.histplot(data[feature], bins=40, kde=True)
    plt.title(feature)

plt.tight_layout()
plt.show()