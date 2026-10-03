import streamlit as st
import numpy as np

st.set_page_config(
    page_title="CATGUARD",
    page_icon="C",
    layout="wide"
)

# Load trained model
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

    # Apply training-time scaling
    input_scaled = (input_data - mean) / std

    # Add bias
    input_scaled = np.insert(input_scaled, 0, 1)

    probability = sigmoid(input_scaled @ weights)

    return probability


# -------------------------
# Header
# -------------------------

st.title("CATGUARD")
st.subheader("AI-Powered Equipment Health & Failure Prediction")

st.write(
    "Enter machine operating conditions to estimate the probability "
    "of machine failure."
)

st.divider()


# -------------------------
# Input section
# -------------------------

st.header("Machine Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=320.0,
        value=300.0,
        step=0.1
    )

with col2:
    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=330.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

with col3:
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


st.divider()


# -------------------------
# Prediction
# -------------------------

if st.button("Analyze Machine", type="primary"):

    probability = predict(
        machine_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    )

    probability_percent = probability * 100

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Failure Probability",
            f"{probability_percent:.2f}%"
        )

    with result_col2:

        if probability >= 0.5:
            st.error("HIGH RISK: Potential Machine Failure")
        else:
            st.success("NORMAL: Low Failure Probability")

    st.progress(float(probability))

    st.subheader("Analyzed Parameters")

    st.write({
        "Machine Type": machine_type,
        "Air Temperature (K)": air_temperature,
        "Process Temperature (K)": process_temperature,
        "Rotational Speed (RPM)": rotational_speed,
        "Torque (Nm)": torque,
        "Tool Wear (minutes)": tool_wear
    })

    st.info(
        "This is a student predictive-maintenance prototype "
        "built using historical machine sensor data."
    )
