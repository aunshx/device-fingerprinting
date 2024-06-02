'''
Purpose:
This script reads a JSON file containing a list of objects, extracts operationOutput and platform, 
and computes Dynamic Time Warping (DTW) distances. The computed distance matrices are saved for later use.
'''

import json
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import os

# Function to extract the required fields
def extract_fields(item):
    return {
        "entryId": item["entryId"],
        "operationOutput": np.fromstring(item["operationOutput"].strip('[]'), sep=','),
        "platform": item["platform"]
    }

# Function to sample 100 entries from each platform
def sample_entries(data, platforms, num_samples=100, random_state=42):
    np.random.seed(random_state)  # Set the seed for reproducibility
    sampled_data = []
    for platform in platforms:
        platform_data = [item for item in data if item["platform"] == platform]
        sampled_data.extend(np.random.choice(platform_data, num_samples, replace=False))
    return sampled_data

input_file = './data/formatted_data_for_dtw.json'

train_distance_matrix_file = './weights/dtw/train_distance_matrix.npy'
test_distance_matrix_file = './weights/dtw/test_distance_matrix.npy'
label_file = './weights/dtw/labels.npy'
train_indices_file = './weights/dtw/train_indices.npy'
test_indices_file = './weights/dtw/test_indices.npy'

# Read the entire JSON file into memory
print("Read input data...")
with open(input_file, 'r') as infile:
    data = json.load(infile)

# Extract the necessary fields
print("Preprocessing data...")
processed_data = [extract_fields(item) for item in tqdm(data, desc="Preprocessing items")]

# Get unique platforms
platforms = list(set(item["platform"] for item in processed_data))

# Sample 100 entries from each platform
print("Sampling entries...")
sampled_data = sample_entries(processed_data, platforms, num_samples=25, random_state=42)

# Prepare the data for model training
X = [item["operationOutput"] for item in sampled_data]
y = [item["platform"] for item in sampled_data]

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y_encoded, range(len(X)), test_size=0.2, random_state=42)

# Compute DTW distance for a pair of sequences
def compute_dtw_distance(x1, x2):
    return dtw.distance(x1, x2)

# Function to compute DTW distance matrix in parallel
def dtw_distance_matrix(X, desc="Computing DTW distances", n_jobs=8):
    size = len(X)
    matrix = np.zeros((size, size))
    
    for i in tqdm(range(size), desc=desc):
        distances = Parallel(n_jobs=n_jobs)(delayed(compute_dtw_distance)(X[i], X[j]) for j in range(i, size))
        for j, distance in enumerate(distances):
            matrix[i, i+j] = distance
            matrix[i+j, i] = distance
    
    return matrix

print("Computing DTW distances for training set...")
train_distance_matrix = dtw_distance_matrix(X_train, desc="Training set DTW")

print("Computing DTW distances for the test set...")
test_distance_matrix = np.zeros((len(X_test), len(X_train)))
for i in tqdm(range(len(X_test)), desc="Computing test DTW distances"):
    distances = Parallel(n_jobs=8)(delayed(compute_dtw_distance)(X_test[i], X_train[j]) for j in range(len(X_train)))
    for j, distance in enumerate(distances):
        test_distance_matrix[i, j] = distance

# Save the distance matrices and labels
print("Saving distance matrices and labels...")
np.save(train_distance_matrix_file, train_distance_matrix)
np.save(test_distance_matrix_file, test_distance_matrix)
np.save(label_file, y_encoded)
np.save(train_indices_file, train_indices)
np.save(test_indices_file, test_indices)

print("Processing complete!")
