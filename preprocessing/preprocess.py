import json
import os

def split_json_file(file_path, chunk_size, output_folder='chunks'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            chunk_file_path = os.path.join(output_folder, f'chunk_{idx}.json')
            with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
                json.dump(chunk, chunk_file, indent=4)

# Example usage
split_json_file('../data/HASHED_combined_data_upto_oct_31_2023.json', 1000)