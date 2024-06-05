import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# Load the processed data from JSON file
with open('ml/ml_analysis/all/data_processed(all).json') as file:
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

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'lambda': [0, 1, 10],  # L2 regularization term
    'alpha': [0, 1, 10]     # L1 regularization term
}

xgb = XGBClassifier(objective='multi:softmax', num_class=len(platform_categories), random_state=42, verbosity=0)
grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_xgb = grid_search.best_estimator_

# Save the model
joblib.dump(best_xgb, 'xgboost_model(all).pkl')
print("Model saved as 'xgboost_model(all).pkl'")

# Make predictions
y_pred_train = best_xgb.predict(X_train)
y_pred_test = best_xgb.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print classification report
print("Classification Report for Training Data:")
print(classification_report(y_train, y_pred_train))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_pred_test))