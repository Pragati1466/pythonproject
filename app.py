import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load

# Load the trained model and preprocessing pipeline
model = tf.keras.models.load_model('cybersecurity_model.h5')
preprocessor = load('preprocessing_pipeline.joblib')

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
        [[sensor_data, vehicle_speed, network_traffic, sensor_type, sensor_status, vehicle_model, firmware_version, geofencing_status]],
        columns=['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic', 'Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
    )
    
    # Preprocess the input data
    input_data_processed = preprocessor.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_processed)
    
    # Display the result
    if prediction[0] > 0.5:
        st.markdown('### High Probability of Adversarial Attack')
    else:
        st.markdown('### Low Probability of Adversarial Attack')

