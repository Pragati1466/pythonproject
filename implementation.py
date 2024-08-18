import pandas as pd
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.classifiers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
csv_file_path = '/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows to understand the data structure
print(data.head())

# Feature columns and label column
features = ['Sensor_Data', 'Attack_Type', 'Vehicle_Speed', 'Sensor_Type', 'Attack_Severity', 'Attack_Duration', 'Attack_Frequency']
label = 'Response_Action'

# Ensure the feature and label columns exist in the dataset
if not all(col in data.columns for col in features + [label]):
    raise ValueError("One or more feature or label columns are missing in the dataset.")

# Preprocess data
X = data[features]
y = data[label]

# Convert categorical features to numeric if necessary
X = pd.get_dummies(X)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and compile the TensorFlow model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('cybersecurity_model.h5')  # Save the model to use in app.py

# Wrap your TensorFlow model with ART
classifier = KerasClassifier(model=model)

# Define attack
attack = FastGradientMethod(classifier, eps=0.1)

# Generate adversarial examples
X_test_adv = attack.generate(X_test)

# Test the model on adversarial examples
accuracy = model.evaluate(X_test_adv, y_test)
print(f'Accuracy on adversarial examples: {accuracy[1]*100:.2f}%')
