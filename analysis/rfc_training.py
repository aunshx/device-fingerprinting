import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load data from a JSON file (ensure the file contains 56,000 entries)
print("Loading data...")
with open('./data/formatted_data_for_dtw.json', 'r') as file:
    data = json.load(file)

# get only 1000 entries of each platform
platforms = ['Linux', 'Windows', 'iOS', 'macOS']

new_data = []
for platform in platforms:
    platform_data = [item for item in data if item["platform"] == platform]
    new_data.extend(platform_data[:1000])

print(f"Total entries: {len(new_data)}")
data = new_data

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Define the fixed length for padding/truncation
fixed_length = 263

def pad_or_truncate(arr, length):
    if len(arr) > length:
        return arr[:length]
    elif len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'constant')
    return arr

# Apply padding/truncation to each operationOutput
df['operationOutput'] = df['operationOutput'].apply(lambda x: pad_or_truncate(x, fixed_length))

# Extract statistical features
df['mean'] = df['operationOutput'].apply(np.mean)
df['std'] = df['operationOutput'].apply(np.std)
df['min'] = df['operationOutput'].apply(np.min)
df['max'] = df['operationOutput'].apply(np.max)

# Create feature matrix and labels
print("Creating feature matrix and labels...")
X = np.vstack(df['operationOutput'].values)
X = np.hstack([X, df[['mean', 'std', 'min', 'max']].values])
y = df['platform']

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Classes: {label_encoder.classes_}")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter tuning
print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [800],
    # 'max_depth': [None, 10, 20, 30],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
}

model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use RandomizedSearchCV for a more efficient search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")
# Save the trained model
print("Saving the model...")
joblib.dump(best_model, './weights/rfc/random_forest_model.pkl')
joblib.dump(label_encoder, './weights/rfc/label_encoder.pkl')
joblib.dump(scaler, './weights/rfc/scaler.pkl')

# Model evaluation
print("Evaluating the model...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Model training and evaluation complete!")
