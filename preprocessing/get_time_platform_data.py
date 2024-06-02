'''
Purpose:
This script reads a JSON file containing a list of objects, extracts specific fields from each object, 
and writes the extracted data to a new JSON file. The script includes a progress bar to indicate the 
processing progress.
'''

import json
from tqdm import tqdm
import numpy as np

# Determine the plaform of the data
def determine_platform(platform):
    if 'Linux' in platform:
        return 'Linux'
    elif 'Win' in platform:
        return 'Windows'
    elif 'iPad' in platform or 'iPhone' in platform:
        return 'iOS'
    elif 'Mac' in platform:
        return 'macOS'
    else:
        return 'Other'

# Function to extract the required fields
def extract_fields(item):
    return {
        "entryId": item["entryId"],
        "operationOutput": np.fromstring(item["operationOutput"].strip('[]'), sep=',').tolist(),
        "platform": determine_platform(item["components"]["platform"]["value"])
    }

input_file = './data/extracted_and_reformatted_data.json'
output_file = './data/formatted_data_for_dtw.json'

# Read the entire JSON file into memory
print("Read input data...")
with open(input_file, 'r') as infile:
    data = json.load(infile)

new_data = []

# Process each item and add to new_data
print("Processing data...")
for item in tqdm(data, desc="Processing items"):
    new_data.append(extract_fields(item))

# Write the new data to a new JSON file
print("Write output data...")
with open(output_file, 'w') as outfile:
    json.dump(new_data, outfile, indent=4)

print("Processing complete!")
