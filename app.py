import streamlit as st
import pandas as pd
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.classifiers import KerasClassifier
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your model
model = load_model('cybersecurity_model.h5')

# Load your dataset
csv_file_path = '/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv'
data = pd.read_csv(csv_file_path)

# Define features
features = ['Sensor_Data', 'Attack_Type', 'Vehicle_Speed', 'Sensor_Type', 'Attack_Severity', 'Attack_Duration', 'Attack_Frequency']
label = 'Response_Action'

# Preprocess the data
X = data[features]
y = data[label]

# Convert categorical features to numeric if necessary
X = pd.get_dummies(X)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Streamlit for interaction
st.title("Cybersecurity in Autonomous Vehicles")
st.write("Enhancing Cybersecurity through Adversarial Robustness")

# User input for new data points
input_data = []
for feature in features:
    value = st.text_input(f'Enter {feature}', '0')
    input_data.append(float(value))

# Convert to numpy array and reshape for model input
input_data = np.array(input_data).reshape(1, -1)

# Normalize the input
input_data_scaled = scaler.transform(input_data)

# Predict using the model
prediction = model.predict(input_data_scaled)
st.write(f'Predicted Response Action: {prediction[0]}')

# Generate adversarial example
if st.button('Generate Adversarial Example'):
    classifier = KerasClassifier(model=model)
    attack = FastGradientMethod(classifier, eps=0.1)
    adversarial_example = attack.generate(input_data_scaled)
    adversarial_prediction = model.predict(adversarial_example)
    st.write(f'Adversarial Example Prediction: {adversarial_prediction[0]}')
