import json

def extract_entry_components(input_filename, output_filename):
    # Load JSON data from a file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    # Create a list to hold the extracted data
    extracted_data = []

    for entry in data:
        entryId = entry[0]
        # Extract 'components' from the JSON string in the third element
        components_json_str = entry[2]
        components_dict = json.loads(components_json_str.replace("'", '"'))
        components = components_dict['fingerprint']['components']

        extracted_data.append({'entryId': entryId, 'components': components})

    # Write the extracted data to a JSON file
    with open(output_filename, 'w') as jsonfile:
        json.dump(extracted_data, jsonfile, indent=4)

    print(f"Data has been extracted and saved to '{output_filename}'")

# Example usage
extract_entry_components('./data/HASHED_combined_data_upto_oct_31_2023.json', './data/extracted_component_data.json')
