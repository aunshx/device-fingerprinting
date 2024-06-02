import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Define file paths
train_distance_matrix_file = './weights/dtw/train_distance_matrix.npy'
test_distance_matrix_file = './weights/dtw/test_distance_matrix.npy'
label_file = './weights/dtw/labels.npy'
train_indices_file = './weights/dtw/train_indices.npy'
test_indices_file = './weights/dtw/test_indices.npy'

# Load the distance matrices and labels
print("Loading distance matrices and labels...")
train_distance_matrix = np.load(train_distance_matrix_file)
test_distance_matrix = np.load(test_distance_matrix_file)
y_encoded = np.load(label_file)
train_indices = np.load(train_indices_file)
test_indices = np.load(test_indices_file)

# Prepare training and testing labels
y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]

# Train KNN classifier
print("Training KNN classifier...")
knn = KNeighborsClassifier(metric='precomputed')
knn.fit(train_distance_matrix, y_train)

# Predict labels for the test set
print("Predicting test set labels...")
y_pred = knn.predict(test_distance_matrix)

# Evaluate the classifier
print("Evaluating the classifier...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
