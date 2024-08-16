from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('cybersecurity_model.h5')
except Exception as e:
    raise RuntimeError(f'Error loading model: {e}')

# Load the scaler used during training
try:
    scaler = load('scaler.joblib')
except Exception as e:
    raise RuntimeError(f'Error loading scaler: {e}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON data from the request
        data = request.json

        # Define expected features
        expected_features = [
            'Sensor_Data', 'Vehicle_Speed', 'Location', 'Sensor_Type', 'Sensor_Status',
            'Attack_Severity', 'Attack_Duration', 'Attack_Frequency', 'Vehicle_Model',
            'Firmware_Version', 'Network_Traffic', 'Error_Code', 'Geofencing_Status'
        ]

        # Convert JSON features to DataFrame
        features = pd.DataFrame(data['features'], index=[0])

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in features.columns:
                features[feature] = 0  # Fill missing columns with default value

        # Reorder columns to match the model's input
        features = features[expected_features]

        # Scale features using the same scaler
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
