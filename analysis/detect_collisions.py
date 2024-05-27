import csv
from collections import defaultdict

def detect_collisions(file_path):
    hash_counts = defaultdict(int)
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hashed_string = row['hashed_string']
            hash_counts[hashed_string] += 1

    total_hashes = sum(hash_counts.values())
    unique_hashes = len(hash_counts)
    collisions = sum(count - 1 for count in hash_counts.values() if count > 1)
    collision_rate = collisions / total_hashes * 100 if total_hashes else 0

    print(f"Total hashes processed: {total_hashes}")
    print(f"Unique hashes: {unique_hashes}")
    print(f"Collisions detected: {collisions}")
    print(f"Collision rate: {collision_rate:.2f}%")
