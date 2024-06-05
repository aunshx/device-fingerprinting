import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('xgboost_model(win_mac).pkl')

# Define a function to extract features from operation output
def extract_features(operation_output):
    features = [
        np.mean(operation_output),
        np.median(operation_output),
        np.std(operation_output),
        np.min(operation_output),
        np.max(operation_output)
    ]
    return np.array(features).reshape(1, -1)  # Reshape to a 2D array (1 sample)

# Load the test JSON file
json_file_path = 'ml/ml_analysis/win_mac/win_mac_test.json'

# try:
with open(json_file_path, 'r') as file:
    test_data = json.load(file)
# except json.JSONDecodeError as e:
#     print(f"Error decoding JSON: {e}")
#     raise
# Define platform categories
platform_categories = {
    0: 'MacIntel',
    1: 'Win32'
}

# Lists to store actual and predicted platforms
actual_platforms = []
predicted_platforms = []

# Iterate through each entry in the test data
for entry in test_data:
    actual_platform = entry['platform']
    operation_output_str = entry['operationOutput']
    # Convert the example output string to a list of floats
    operation_output_str = operation_output_str.strip('[]')
    example_output = list(map(float, operation_output_str.split(',')))
    # Extract features from the example operation output
    example_features = extract_features(example_output)
    # Predict the platform for the example operation output
    predicted_platform = model.predict(example_features)[0]
    # Map the predicted platform code to its actual category
    predicted_platform_name = platform_categories[predicted_platform]
    # Store actual and predicted platforms
    actual_platforms.append(actual_platform)
    predicted_platforms.append(predicted_platform_name)

# Plotting the actual versus predicted platforms
plt.figure(figsize=(12, 8))
index = np.arange(len(test_data))
plt.scatter(index, actual_platforms, marker='o', label='Actual')
plt.scatter(index, predicted_platforms, marker='x', label='Predicted', color='red')
plt.title('Actual vs Predicted Platforms')
plt.xlabel('Entry/Index')
plt.ylabel('Platform')
plt.xticks(index)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a DataFrame to store the predicted and actual platforms
platform_df = pd.DataFrame({'Actual': actual_platforms, 'Predicted': predicted_platforms})

# Save the DataFrame to a CSV file
#platform_df.to_csv('csv_knn_win_mac.csv', index=False)
