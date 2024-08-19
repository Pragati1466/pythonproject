import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load a sample dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define sample data for prediction (using the first sample from X_test as an example)
sample_data = X_test[0].reshape(1, -1)

# Number of times to repeat the prediction to get an average
n_iterations = 1000
times = []

# Measure response times
for _ in range(n_iterations):
    start_time = time.time()
    _ = model.predict(sample_data)
    end_time = time.time()

    response_time = end_time - start_time
    times.append(response_time)

# Calculate the average response time
average_response_time = np.mean(times)

print(f"Average response time over {n_iterations} iterations: {average_response_time:.6f} seconds")
