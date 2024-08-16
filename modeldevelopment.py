import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from joblib import dump

# Load your dataset
df = pd.read_csv('/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv')

# Define your features and labels
features = df.drop(columns=['Adversarial_Attack'])  # Assuming 'Adversarial_Attack' is your label
labels = df['Adversarial_Attack']

# Define categorical and numerical features
categorical_features = [
    'Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status'
]
numerical_features = [
    'Sensor_Data', 'Vehicle_Speed', 'Network_Traffic'
]

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Preprocess the features
X = pipeline.fit_transform(features)

# Convert labels to numpy array
y = np.array(labels)

# Save the preprocessing pipeline
dump(pipeline, 'preprocessing_pipeline.joblib')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Adjust input shape based on X_train
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Save the model to file
model.save('cybersecurity_model.h5')
