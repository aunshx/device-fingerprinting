import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from dtaidistance import dtw
import joblib
import json

# Load data from a JSON file
print("Loading data...")
with open('./data/formatted_data_for_dtw.json', 'r') as file:
    data = json.load(file)
data = data[:1000]

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

# Create feature matrix and labels
print("Creating feature matrix and labels...")
X = np.vstack(df['operationOutput'].values)
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

# Custom DTW distance function for k-NN
def dtw_distance(x, y):
    return dtw.distance(x, y)

# k-NN classifier with DTW distance
print("Training the k-NN model with DTW...")
knn = KNeighborsClassifier(n_neighbors=3, metric=dtw_distance, n_jobs=-1)
knn.fit(X_train, y_train)

# Model evaluation
print("Evaluating the model...")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and preprocessors
print("Saving the model and preprocessors...")
joblib.dump(knn, './weights/knn/dtw_knn_model.pkl')
joblib.dump(label_encoder, './weights/knn/dtw_knn_label_encoder.pkl')
joblib.dump(scaler, './weights/knn/dtw_knn_scaler.pkl')

print("Model training and evaluation complete!")
