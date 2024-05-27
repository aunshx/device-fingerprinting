import hashlib
from preprocessing.hash_entry_components import extract_and_save_entry_components

def hash_function(value, algorithm):
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()

input_filename = 'data/HASHED_combined_data_upto_oct_31_2023.json'

# MD5 (128-bit)
extract_and_save_entry_components(
    input_filename=input_filename,
    output_filename='data/md5_hashed_components.csv',
    hash_function=lambda x: hash_function(x, 'md5'),
    chunk_size=1000
)

# # SHA-1 (160-bit)
# extract_and_save_entry_components(
#     input_filename=input_filename,
#     output_filename='data/sha1_hashed_components.csv',
#     hash_function=lambda x: hash_function(x, 'sha1'),
#     chunk_size=1000
# )

# # SHA-224 (224-bit)
# extract_and_save_entry_components(
#     input_filename=input_filename,
#     output_filename='data/sha224_hashed_components.csv',
#     hash_function=lambda x: hash_function(x, 'sha224'),
#     chunk_size=1000
# )

# # SHA-256 (256-bit)
# extract_and_save_entry_components(
#     input_filename=input_filename,
#     output_filename='data/sha256_hashed_components.csv',
#     hash_function=lambda x: hash_function(x, 'sha256'),
#     chunk_size=1000
# )

# # SHA-512 (512-bit)
# extract_and_save_entry_components(
#     input_filename=input_filename,
#     output_filename='data/sha512_hashed_components.csv',
#     hash_function=lambda x: hash_function(x, 'sha512'),
#     chunk_size=1000
# )

# # blake2b (512-bit)
# extract_and_save_entry_components(
#     input_filename=input_filename,
#     output_filename='data/blake2b_hashed_components.csv',
#     hash_function=lambda x: hash_function(x, 'blake2b'),
#     chunk_size=1000
# )