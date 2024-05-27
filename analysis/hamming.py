import csv
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count

def hamming_distance(args):
    str1, str2 = args
    assert len(str1) == len(str2), "Strings must be of the same length"
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def process_chunk(chunk):
    hamming_distances = []
    for hash1, hash2 in combinations(chunk, 2):
        if len(hash1) == len(hash2):
            hamming_distances.append(hamming_distance((hash1, hash2)))
    return hamming_distances

def detect_hamming_distances(file_path, chunk_size=1000):
    hash_counts = defaultdict(int)
    hashes = []

    print(f"Reading hashed strings from '{file_path}'...")
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hashed_string = row['hashed_string']
            hash_counts[hashed_string] += 1
            hashes.append(hashed_string)

    chunked_hamming_distances = []
    
    print(f"Calculating Hamming distances for {len(hashes)} hashes...")
    # Process in chunks
    for i in range(0, len(hashes), chunk_size):
        chunk = hashes[i:i + chunk_size]
        with Pool(cpu_count()) as pool:
            chunk_hamming_distances = pool.map(process_chunk, [chunk])
            for distances in chunk_hamming_distances:
                chunked_hamming_distances.extend(distances)

    if chunked_hamming_distances:
        min_hamming = min(chunked_hamming_distances)
        max_hamming = max(chunked_hamming_distances)
        avg_hamming = sum(chunked_hamming_distances) / len(chunked_hamming_distances)

        print(f"Minimum Hamming distance: {min_hamming}")
        print(f"Maximum Hamming distance: {max_hamming}")
        print(f"Average Hamming distance: {avg_hamming:.2f}")
    else:
        print("No valid pairs for Hamming distance calculation (hashes of different lengths).")

# Example usage
if __name__ == "__main__":
    detect_hamming_distances('./data/sha256_hashed_components.csv', chunk_size=1000)
