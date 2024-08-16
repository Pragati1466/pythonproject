import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from joblib import dump

# Load the dataset
csv_file_path = '/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_data_50_rows.csv'  # Update this path to your CSV file
data = pd.read_csv(csv_file_path)

# Display the first few rows to understand the data structure
print(data.head())

# Define features and label
features = ['Sensor_Data', 'Vehicle_Speed', 'Location', 'Sensor_Type', 'Sensor_Status',
             'Attack_Severity', 'Attack_Duration', 'Attack_Frequency', 'Vehicle_Model',
             'Firmware_Version', 'Network_Traffic', 'Error_Code', 'Geofencing_Status']
label = 'Adversarial_Attack'

# Ensure that there are no missing values in the features and label
if data[features].isnull().sum().any() or data[label].isnull().sum() > 0:
    print("Data contains missing values. Handling missing values...")
    data = data.dropna()  # Example handling, adjust as needed

# Separate features and label
X = data[features]
y = data[label]

# Encode categorical features
# Convert all categorical features to string if they are not
categorical_features = ['Location', 'Sensor_Type', 'Sensor_Status', 'Vehicle_Model', 'Firmware_Version', 'Geofencing_Status']
for feature in categorical_features:
    X[feature] = X[feature].astype(str)

# Encode categorical features using one-hot encoding
X = pd.get_dummies(X, columns=categorical_features)

# Convert label to numeric values if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
dump(scaler, 'scaler.joblib')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verify the shapes of the datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Optionally save the preprocessed data if needed
# pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
# pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
# pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
