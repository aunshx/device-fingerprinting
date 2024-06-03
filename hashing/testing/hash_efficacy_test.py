import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import time
import hashlib
import os

# Function to compute Hamming distance between two hexadecimal strings
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Paths to the CSV files
file_paths = [
    '../data/sha224_hashed_components.csv',
    '../data/sha512_hashed_components.csv',
    '../data/md5_hashed_components.csv',
    '../data/blake2b_hashed_components.csv',
    '../data/sha1_hashed_components.csv',
    '../data/sha256_hashed_components.csv'
]

# List to collect results for each file
results = []

# Iterate through each file path
for file_path in file_paths:
    data = pd.read_csv(file_path)

    # Extract the hashing algorithm from the file name
    algorithm = os.path.basename(file_path).split('_')[0]

    # Basic statistics
    total_entries = data.shape[0]
    num_unique_hashes = data['hashed_string'].nunique()

    # Check for duplicates and their frequency
    hash_counts = data['hashed_string'].value_counts()
    num_duplicates = (hash_counts > 1).sum()
    duplicate_hashes = hash_counts[hash_counts > 1]

    # Calculate collision rate
    collision_rate = num_duplicates / total_entries

    # Calculate the proportion of unique hashes
    uniqueness_ratio = num_unique_hashes / total_entries

    # Handle edge cases: Check for missing or malformed data
    missing_hashes = data['hashed_string'].isnull().sum()

    # Measure hashing speed
    sample_data = data['hashed_string'].sample(min(1000, len(data)))

    # Calculate pairwise Hamming distances for a sample of hashes
    sample_hashes = data['hashed_string'].sample(min(1000, len(data)))
    hamming_distances = []

    for hash1, hash2 in combinations(sample_hashes, 2):
        hamming_distances.append(hamming_distance(hash1, hash2))

    # Analyze the distribution of Hamming distances
    hamming_distances = np.array(hamming_distances)
    avg_hamming_distance = np.mean(hamming_distances)
    std_hamming_distance = np.std(hamming_distances)

    # Collect the results
    results.append({
        'algorithm': algorithm,
        'total_entries': total_entries,
        'num_unique_hashes': num_unique_hashes,
        'num_duplicates': num_duplicates,
        'collision_rate': collision_rate,
        'uniqueness_ratio': uniqueness_ratio,
        'missing_hashes': missing_hashes,
        'avg_hamming_distance': avg_hamming_distance,
        'std_hamming_distance': std_hamming_distance
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('./data/hash_analysis_results.csv', index=False)