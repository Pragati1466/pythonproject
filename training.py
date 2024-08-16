import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from joblib import dump

# Load and preprocess data
csv_file_path = '/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv'
data = pd.read_csv(csv_file_path)

# Clean 'Vehicle_Speed' column (Convert to float if necessary)
data['Vehicle_Speed'] = data['Vehicle_Speed'].str.extract('(\d+\.?\d*)').astype(float)  # Extract numbers and convert to float

# Clean 'Network_Traffic' column (Convert to float if necessary)
data['Network_Traffic'] = data['Network_Traffic'].str.extract('(\d+\.?\d*)').astype(float)  # Extract numbers and convert to float

# Define features and label (Ensure these match what you plan to use during prediction)
features = ['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic', 'Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
label = 'Adversarial_Attack'

# Separate features and label
X = data[features]
y = data[label]

# Update feature types
categorical_features = ['Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
numerical_features = ['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic']

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler()  # For numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # For categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Preprocess the features
X_processed = pipeline.fit_transform(X)

# Save the entire preprocessing pipeline (this is important for consistency during prediction)
dump(pipeline, 'preprocessing_pipeline.joblib')

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Define and compile model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Ensure input shape matches X_train
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model to file (use the same file name in the Flask app)
model.save('cybersecurity_model.h5')
print("Model saved as cybersecurity_model.h5")
