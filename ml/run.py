import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data from JSON file
with open('./data_current/processed_data.json') as file:
    data = json.load(file)

# Prepare data for the model
platforms = []
features = []

for entry in data:
    platform = entry['platform']
    operation_output_str = entry['operationOutput'].strip('[]').split(', ')
    
    # Convert operationOutput to numerical values
    try:
        operation_output = list(map(float, operation_output_str))
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Train a RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model and platform categories
joblib.dump(clf, 'random_forest_model.pkl')
joblib.dump(platform_categories.tolist(), 'platform_categories.pkl')
print("Model and platform categories saved as 'random_forest_model.pkl' and 'platform_categories.pkl'")

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Function to predict platform from operationOutput with probabilities
def predict_platform_with_prob(operation_output):
    feature = [
        np.mean(operation_output),
        np.median(operation_output),
        np.std(operation_output),
        np.min(operation_output),
        np.max(operation_output)
    ]
    probabilities = clf.predict_proba([feature])[0]
    predicted_index = np.argmax(probabilities)
    predicted_platform = platform_categories[predicted_index]
    probability_percentage = probabilities[predicted_index] * 100
    return predicted_platform, probability_percentage, probabilities

# Example usage - Linux i686
example_output_str ="[333.5150000057183, 411.8500000331551, 1430.1800000248477, 218.84499996667728, 81.52499998686835, 111.66999995475635, 427.89500002982095, 371.99000001419336, 251.97000004118308, 167.40999999456108, 536.5849999943748, 590.0400000391528, 2502.4349999730475, 292.7899999776855, 759.7050000331365, 457.9850000445731, 1391.089999990072, 1423.8449999829754, 1408.8449999690056, 964.4400000106543, 189.24500001594424, 385.8750000363216, 403.05500000249594, 186.54500000411645, 88.82000000448897, 1586.2399999750778, 658.3600000012666, 403.65499997278675, 750.9350000182167, 495.14500005170703, 531.9649999728426, 719.2599999834783, 10515.069999964908, 2559.644999972079, 1547.8899999870919, 15036.550000018906, 115.90000003343448, 77.04499998362735, 884.0400000335649, 562.2200000216253, 3958.344999991823, 384.9400000181049, 539.7349999984726, 455.4249999928288, 3485.4550000163727]"

example_output = list(map(float, example_output_str.strip('[]').split(', ')))

predicted_platform, probability, probabilities = predict_platform_with_prob(example_output)
print(f'Predicted Platform: {predicted_platform} Probability: {probability:.2f}%')

# To print probabilities for all classes
for i, prob in enumerate(probabilities):
    print(f'Platform: {platform_categories[i]} Probability: {prob * 100:.2f}%')
