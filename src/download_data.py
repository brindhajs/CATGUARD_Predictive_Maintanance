from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Downloading AI4I 2020 Predictive Maintenance Dataset...")

# Fetch dataset from UCI
dataset = fetch_ucirepo(id=601)

# Get features and target
X = dataset.data.features
y = dataset.data.targets

# Combine them into one dataframe
df = pd.concat([X, y], axis=1)

# Save the dataset inside our project
df.to_csv("data/ai4i2020.csv", index=False)

print("Dataset downloaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumns:")
print(df.columns.tolist())