import pandas as pd

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

print("=" * 50)
print("CATGUARD DATASET EXPLORATION")
print("=" * 50)

# 1. Dataset size
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Column names
print("\n2. Columns:")
print(df.columns.tolist())

# 3. First five rows
print("\n3. First 5 Rows:")
print(df.head())

# 4. Data types
print("\n4. Data Types:")
print(df.dtypes)

# 5. Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# 6. Machine failure distribution
print("\n6. Machine Failure Distribution:")
print(df["Machine failure"].value_counts())

# 7. Failure percentages
print("\n7. Machine Failure Percentage:")
print(df["Machine failure"].value_counts(normalize=True) * 100)