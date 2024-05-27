import hashlib
from preprocessing.hash_entry_components import extract_and_save_entry_components

def hash_function(value, algorithm):
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()

extract_and_save_entry_components(
    input_filename='data/HASHED_combined_data_upto_oct_31_2023.json',
    output_filename='output/sha256_hashed_components.csv',
    hash_function=lambda x: hash_function(x, 'sha256'),
    chunk_size=1000
)