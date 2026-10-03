import pandas as pd
import numpy as np

# Load prepared data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Convert to numbers
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)

# Feature scaling
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add bias column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Give more importance to rare failure cases
normal_count = np.sum(y_train == 0)
failure_count = np.sum(y_train == 1)

failure_weight = normal_count / failure_count

print("Normal samples:", normal_count)
print("Failure samples:", failure_count)
print("Failure weight:", round(failure_weight, 2))

# Initialize model weights
weights = np.zeros(X_train.shape[1])

learning_rate = 0.01
epochs = 3000

# Train logistic regression
for epoch in range(epochs):

    probability = sigmoid(X_train @ weights)

    error = probability - y_train

    sample_weight = np.where(
        y_train == 1,
        failure_weight,
        1.0
    )

    gradient = (
        X_train.T @ (error * sample_weight)
    ) / len(y_train)

    weights -= learning_rate * gradient

# Test model
test_probability = sigmoid(X_test @ weights)

threshold = 0.5
predictions = (test_probability >= threshold).astype(int)

# Confusion matrix
tn = np.sum((y_test == 0) & (predictions == 0))
fp = np.sum((y_test == 0) & (predictions == 1))
fn = np.sum((y_test == 1) & (predictions == 0))
tp = np.sum((y_test == 1) & (predictions == 1))

accuracy = (tp + tn) / len(y_test)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\nModel Performance")
print("------------------------")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nConfusion Matrix")
print("------------------------")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

# Save model
np.savez(
    "data/catguard_model.npz",
    weights=weights,
    mean=mean,
    std=std
)

print("\nModel saved successfully.")
