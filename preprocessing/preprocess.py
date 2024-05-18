import json

def split_json_file(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            with open(f'chunk_{idx}.json', 'w', encoding='utf-8') as chunk_file:
                json.dump(chunk, chunk_file, indent=4)


split_json_file('../data/HASHED_combined_data_upto_oct_31_2023.json', 1000)
