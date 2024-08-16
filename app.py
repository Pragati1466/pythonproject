from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
import logging

# Set up logging
logging.basicConfig(filename='system.log', level=logging.INFO)

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('cybersecurity_model.h5')  # Ensure correct path
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise

# Load the preprocessing pipeline
try:
    preprocessor = load('preprocessing_pipeline.joblib')  # Ensure correct path
except Exception as e:
    logging.error(f'Error loading preprocessing pipeline: {e}')
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the POST request
        data = request.json
        features = pd.DataFrame(data['features'])

        # Preprocess features
        features_processed = preprocessor.transform(features)

        # Make prediction
        prediction = model.predict(features_processed)

        # Log the request and prediction
        logging.info(f'Features: {features}, Prediction: {prediction.tolist()}')

        # Return prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logging.error(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
