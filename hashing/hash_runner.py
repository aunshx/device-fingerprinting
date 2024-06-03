import hashlib
import time
import pandas as pd
from functions.hash_entry_components import extract_and_save_entry_components

def hash_function(value, algorithm):
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()

input_filename = '../data/HASHED_combined_data_upto_oct_31_2023.json'
hash_algorithms = ['md5', 'sha1', 'sha224', 'sha256', 'sha512', 'blake2b']

time_stats = []

for algorithm in hash_algorithms:
    output_filename = f'./data_2/{algorithm}_hashed_components.csv'
    start_time = time.time()
    
    extract_and_save_entry_components(
        input_filename=input_filename,
        output_filename=output_filename,
        hash_function=lambda x, alg=algorithm: hash_function(x, alg),
        chunk_size=1000
    )
    
    end_time = time.time()
    duration = end_time - start_time
    time_stats.append({'Hash Algorithm': algorithm, 'Time Taken (seconds)': duration})
    print(f"Hash Algorithm, {algorithm}, Time Taken (seconds), {duration}")

# Save time statistics to a CSV file
time_stats_df = pd.DataFrame(time_stats)
time_stats_df.to_csv('../data/hash_time_stats.csv', index=False)
print("Time statistics saved to '../data/hash_time_stats.csv'")