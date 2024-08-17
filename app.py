import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load

# Load the trained model and preprocessing pipeline
model = tf.keras.models.load_model('cybersecurity_model.h5')
preprocessor = load('preprocessing_pipeline.joblib')

def generate_adversarial_examples(model, x, y_true, epsilon=0.1):
    x = tf.convert_to_tensor(x)
    y_true = tf.convert_to_tensor(y_true)

    # Reshape y_true to match the shape of predictions
    y_true = tf.reshape(y_true, (1, 1))

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x, training=False)
        loss = tf.keras.losses.binary_crossentropy(y_true, predictions)
    gradient = tape.gradient(loss, x)
    adversarial_example = x + epsilon * tf.sign(gradient)
    return adversarial_example

# Streamlit UI
st.header('Cybersecurity Threat Prediction')

# User inputs for the features
sensor_data = st.slider('Sensor Data', 0.0, 100.0)
vehicle_speed = st.slider('Vehicle Speed (in km/h)', 0, 200)
network_traffic = st.slider('Network Traffic (in MB)', 0.0, 1000.0)
sensor_type = st.selectbox('Sensor Type', ['Type 1', 'Type 2', 'Type 3'])
sensor_status = st.selectbox('Sensor Status', ['Active', 'Inactive', 'Error'])
vehicle_model = st.selectbox('Vehicle Model', ['Model A', 'Model B', 'Model C'])
firmware_version = st.selectbox('Firmware Version', ['v1.0', 'v2.0', 'v3.0'])
geofencing_status = st.selectbox('Geofencing Status', ['Enabled', 'Disabled'])

if st.button("Predict Threat"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame(
        [[sensor_data, vehicle_speed, network_traffic, sensor_type, sensor_status, vehicle_model, firmware_version,
          geofencing_status]],
        columns=['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic', 'Sensor_Type', 'Sensor_Status', 'Vehicle_Model',
                 'Firmware_Version', 'Geofencing_Status']
    )

    # Preprocess the input data
    input_data_processed = preprocessor.transform(input_data)

    # Ensure input_data_processed has shape (1, 16) for the model
    input_data_processed = np.expand_dims(input_data_processed, axis=0)

    # Assuming binary classification with a positive label
    y_true = np.array([1])

    try:
        # Generate adversarial example
        input_data_processed_adv = generate_adversarial_examples(model, input_data_processed, y_true)

        # Ensure the data is in batch format (add batch dimension)
        input_data_processed_adv = np.expand_dims(input_data_processed_adv, axis=0)

        # Make a prediction
        prediction = model.predict(input_data_processed_adv)

        # Display the result
        if prediction[0] > 0.5:
            st.markdown('### High Probability of Adversarial Attack')
        else:
            st.markdown('### Low Probability of Adversarial Attack')
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
