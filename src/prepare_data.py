import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/ai4i2020.csv")

print("Original dataset shape:", df.shape)

# Convert machine type into separate yes/no columns
df = pd.get_dummies(df, columns=["Type"], drop_first=True, dtype=int)

features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Type_L",
    "Type_M"
]

target = "Machine failure"

X = df[features]
y = df[target]

print("\nFeatures used by the model:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the prepared datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nCSV files saved successfully.")
