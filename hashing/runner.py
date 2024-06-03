import hashlib
import time
import pandas as pd

def hash_function(value, algorithm):
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()

def measure_hashing_speed(data, algorithm):
    start_time = time.time()
    hashes = [hash_function(value, algorithm) for value in data]
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_hash = total_time / len(data)
    print(f"Total Time for {algorithm}: {total_time:.4f} seconds")
    print(f"Average Time per Hash for {algorithm}: {avg_time_per_hash:.6f} seconds")

    # Save the results to a CSV file
    result_df = pd.DataFrame({
        'data': data,
        'hash': hashes,
        'total_time': [total_time] * len(data),
        'avg_time_per_hash': [avg_time_per_hash] * len(data)
    })
    result_df.to_csv(f'{algorithm}_hashing_speed.csv', index=False)

    return hashes, total_time, avg_time_per_hash

def main():
    data_sample = ['sample1', 'sample2', 'sample3']  # Replace with actual data
    algorithms = ['md5', 'sha1', 'sha224', 'sha256']
    for algorithm in algorithms:
        print(f"\n--- {algorithm.upper()} ---")
        measure_hashing_speed(data_sample, algorithm)

if __name__ == "__main__":
    main()
