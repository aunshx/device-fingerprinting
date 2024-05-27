# This generates a cv file with the component data set to a concatenated string
import json
import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def extract_and_concatenate_components(entry):
    entryId = entry[0]
    components_str = entry[2]
    try:
        components = ast.literal_eval(components_str)["fingerprint"]["components"]
        concatenated_string = []

        for component, details in components.items():
            value = details.get('value', "")
            duration = details.get('duration', 0)

            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            elif isinstance(value, list):
                value = ','.join(map(str, value))
            else:
                value = str(value)

            # Repeat the value based on the duration
            concatenated_string.append(value * duration)

        return {'entryId': entryId, 'concatenated_string': ''.join(concatenated_string)}
    except (SyntaxError, ValueError) as e:
        print(f"Error processing components for entryId {entryId}: {e}")
        return None

def read_json_in_chunks(file, chunk_size=1000):
    """Read a large JSON file in chunks."""
    with open(file, 'r') as f:
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

def extract_and_save_entry_components(input_filename, output_filename, chunk_size=1000):
    print(f"Processing data from '{input_filename}' in chunks of {chunk_size}...")
    num_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    print(f"Using {num_workers} threads to extract and concatenate components")

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['entryId', 'concatenated_string']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data_chunk in read_json_in_chunks(input_filename, chunk_size):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(extract_and_concatenate_components, entry) for entry in data_chunk]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        writer.writerow(result)
            
    
    print(f"Results have been saved to '{output_filename}'")

# Example usage
extract_and_save_entry_components('./data/HASHED_combined_data_upto_oct_31_2023.json', './data/extracted_concatenated_components.csv', chunk_size=500)
