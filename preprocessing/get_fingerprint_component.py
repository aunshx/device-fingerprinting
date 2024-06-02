import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def extract_components(entry):
    entryId = entry[0]
    components_str = entry[2]
    try:
        # Use ast.literal_eval to safely evaluate the string as a dictionary
        components = ast.literal_eval(components_str)["fingerprint"]["components"]
        return {'entryId': entryId, 'browserData': entry[1], 'operationDetails': entry[3], 'operationOutput': entry[4], 'components': components}
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing components for entryId {entryId}: {e}")
        return None

def extract_entry_components(input_filename, output_filename):
    # Load JSON data from a file
    print(f"Loading data from '{input_filename}'...")
    with open(input_filename, 'r') as file:
        data = json.load(file)
    print(f"Loaded {len(data)} entries")

    # Determine the number of threads to use
    num_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    print(f"Using {num_workers} threads to extract components")

    # Use ThreadPoolExecutor to process entries in parallel
    print("Extracting components from entries...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(extract_components, entry) for entry in data]
        extracted_data = []
        for future in as_completed(futures):
            result = future.result()
            if result:
                extracted_data.append(result)
    print(f"Extracted components from {len(extracted_data)} entries")

    # Write the extracted data to a JSON file
    print(f"Saving extracted data to '{output_filename}'...")
    with open(output_filename, 'w') as jsonfile:
        json.dump(extracted_data, jsonfile, indent=4)
    print(f"Data has been extracted and saved to '{output_filename}'")

# Example usage
extract_entry_components('../data/HASHED_combined_data_upto_oct_31_2023.json', '../data/extracted_component_data.json')
