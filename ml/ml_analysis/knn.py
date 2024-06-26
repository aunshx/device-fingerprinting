import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# Load the processed data from JSON file
with open('/Users/saishashetty/Desktop/device-fingerprinting/ml/ml_analysis/win_mac/win_mac.json') as file:
    data = json.load(file)

# Prepare data for the model
platforms = []
features = []

for entry in data:
    platform = entry['platform']
    operation_output_str = entry['operationOutput'].strip('[]').split(', ')
    
    # Convert operationOutput to numerical values
    try:
        operation_output = np.array(list(map(float, operation_output_str)))
    except ValueError as e:
        print(f"Error converting operationOutput to float: {e}")
        continue
    
    # Extract statistical features from operationOutput
    feature = [
        np.mean(operation_output),
        np.median(operation_output),
        np.std(operation_output),
        np.min(operation_output),
        np.max(operation_output)
    ]
    platforms.append(platform)
    features.append(feature)

# Create DataFrame
df = pd.DataFrame(features, columns=['mean', 'median', 'std', 'min', 'max'])
df['platform'] = platforms

# Encode target variable
platform_categories = df['platform'].astype('category').cat.categories
df['platform'] = df['platform'].astype('category').cat.codes

# Split data into training and testing sets
X = df.drop('platform', axis=1)
y = df['platform']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
k = 2  # Adjust k as needed
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the model
joblib.dump(knn, 'knn_model(win_mac).pkl')
print("KNN Model saved as 'knn_model(win_mac).pkl'")

# Print classification report
print("Classification Report for Training Data:")
print(classification_report(y_train, y_pred_train))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_pred_test))