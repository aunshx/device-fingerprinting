import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import json

# Load data from a JSON file
print("Loading data...")
with open('./data/formatted_data_for_dtw.json', 'r') as file:
    data = json.load(file)
# data = data[:1000]  # Limit the data size for quicker testing

# Get 1000 entries of each platform
new_data = []
for platform in ['Linux', 'Windows', 'iOS', 'macOS']:
    platform_data = [item for item in data if item["platform"] == platform]
    new_data.extend(platform_data[:1000])
data = new_data


# print out the numer of entries per platform
for platform in ['Linux', 'Windows', 'iOS', 'macOS']:
    platform_data = [item for item in data if item["platform"] == platform]
    print(f"Number of entries for {platform}: {len(platform_data)}")

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
print("Applying padding/truncation...")
df['operationOutput'] = df['operationOutput'].apply(lambda x: pad_or_truncate(x, fixed_length))

# Extract statistical features
print("Extracting statistical features...")
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

# Train an SVM classifier
print("Training the SVM model...")
svm = SVC(kernel='poly', degree=100, C=1, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# Model evaluation
print("Evaluating the model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and preprocessors
print("Saving the model and preprocessors...")
joblib.dump(svm, './weights/svm/svm_model.pkl')
joblib.dump(label_encoder, './weights/svm/svm_label_encoder.pkl')
joblib.dump(scaler, './weights/svm/svm_scaler.pkl')

print("Model training and evaluation complete!")
