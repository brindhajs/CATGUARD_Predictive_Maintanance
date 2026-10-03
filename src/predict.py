import numpy as np

model = np.load("data/catguard_model.npz")

weights = model["weights"]
mean = model["mean"]
std = model["std"]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):

    # Convert machine type into the same format used during training
    if machine_type == "L":
        type_l = 1
        type_m = 0
    elif machine_type == "M":
        type_l = 0
        type_m = 1
    else:
        type_l = 0
        type_m = 0

    input_data = np.array([
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        type_l,
        type_m
    ], dtype=float)

    # Apply the same scaling used during training
    input_scaled = (input_data - mean) / std

    # Add bias
    input_scaled = np.insert(input_scaled, 0, 1)

    probability = sigmoid(input_scaled @ weights)

    prediction = int(probability >= 0.5)

    return probability, prediction


if __name__ == "__main__":

    probability, prediction = predict(
        "M",
        300,
        310,
        1500,
        40,
        100
    )

    print("Failure probability:", round(probability * 100, 2), "%")
    print("Prediction:", "FAILURE" if prediction == 1 else "NORMAL")
