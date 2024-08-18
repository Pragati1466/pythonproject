import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset with tab delimiter
data = pd.read_csv('test_data.csv', delimiter='\t')

# Print column names to verify
print(data.columns)

# Update column names based on the output
X = data.drop('Adversarial_Attack', axis=1, errors='ignore')  # Drop the target column to get features
y = data['Adversarial_Attack'] if 'Adversarial_Attack' in data.columns else None  # The target variable

if y is None:
    raise ValueError("Target column 'Adversarial_Attack' not found in the dataset.")

# Convert categorical columns to numeric if necessary
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
