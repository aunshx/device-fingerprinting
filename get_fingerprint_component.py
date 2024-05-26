import json
import ast

def extract_entry_components(input_filename, output_filename):
    # Load JSON data from a file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Create a list to hold the extracted data
    extracted_data = []

    for entry in data:
        components_str = entry[2]  # The JSON string with the components
        components = ast.literal_eval(components_str)["fingerprint"]["components"]

    for entry in data:
        entryId = entry[0]
        components_str = entry[2]
        try:
            # Use ast.literal_eval to safely evaluate the string as a dictionary
            components_dict = ast.literal_eval(components_str)
            components = components_dict.get('fingerprint', {}).get('components', {})
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing components for entryId {entryId}: {e}")
            continue

        extracted_data.append({'entryId': entryId, 'components': components})

    # Write the extracted data to a JSON file
    with open(output_filename, 'w') as jsonfile:
        json.dump(extracted_data, jsonfile, indent=4)

    print(f"Data has been extracted and saved to '{output_filename}'")
# Example usage
extract_entry_components('./data/HASHED_combined_data_upto_oct_31_2023.json', './data/extracted_component_data.json')
