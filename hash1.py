import json
import hashlib
import binascii
import ast

def hash_value(value, algorithm):
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()

def hex_to_binary(hex_string):
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

def hamming_distance(hash1, hash2):
    binary1 = hex_to_binary(hash1)
    binary2 = hex_to_binary(hash2)
    return sum(c1 != c2 for c1, c2 in zip(binary1, binary2))

def process_json(json_file):
    print("Loading JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print("Loaded JSON data")
    print("Gathering hashes...")
    hashes = []

    for entry in data:
        # print every 1000 components
        if len(hashes) % 1000 == 0:
            print(len(hashes))
            
        components_str = entry[2]  # The JSON string with the components
        components = ast.literal_eval(components_str)["fingerprint"]["components"]
        
        concatenated_string = ""
        
        for component, details in components.items():
                
            value = details.get('value', "")
            duration = details.get('duration', 0)
            
            # Convert the value to a string, handling different types
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            elif isinstance(value, list):
                value = ','.join(map(str, value))
            else:
                value = str(value)
            
            # Repeat the value according to its duration
            concatenated_string += value * duration

        # Hash the concatenated string
        sha256_hash = hash_value(concatenated_string, 'sha256')
        md5_hash = hash_value(concatenated_string, 'md5')

        hashes.append((sha256_hash, md5_hash))

    print("Calculated hashes")
    
    print("Calculating Hamming distance for SHA256...")
    # Calculate Hamming distances within SHA256 hashes
    sha256_distances = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = hamming_distance(hashes[i][0], hashes[j][0])
            sha256_distances.append(distance)

    print("Calculating Hamming distance for MD5...")
    # Calculate Hamming distances within MD5 hashes
    md5_distances = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = hamming_distance(hashes[i][1], hashes[j][1])
            md5_distances.append(distance)

    # Calculate and print average distances
    avg_sha256_distance = sum(sha256_distances) / len(sha256_distances) if sha256_distances else 0
    avg_md5_distance = sum(md5_distances) / len(md5_distances) if md5_distances else 0

    print(f"Average Hamming Distance within SHA256 hashes: {avg_sha256_distance}")
    print(f"Average Hamming Distance within MD5 hashes: {avg_md5_distance}")

if __name__ == "__main__":
    process_json('./data/HASHED_combined_data_upto_oct_31_2023.json')
