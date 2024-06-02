'''
Purpose:
This script reads a JSON file containing a list of objects, extracts specific fields from each object, 
and writes the extracted data to a new CSV file. The script includes a progress bar to indicate the 
processing progress.

Used this script to determine the main categories of platforms in the data and write the data to a CSV file.
'''

import json
import csv
from tqdm import tqdm

# Function to extract the required fields
def extract_fields(item):
    return {
        "entryId": item["entryId"],
        "platform": item["components"]["platform"]["value"]
    }

input_file = './data/extracted_and_reformatted_data.json'
output_file = './data/platform_data.csv'

print("Read input data...")
# Read the entire JSON file into memory
with open(input_file, 'r') as infile:
    data = json.load(infile)

new_data = []

print("Processing data...")
# Process each item and add to new_data
for item in tqdm(data, desc="Processing items"):
    new_data.append(extract_fields(item))

print("Write output data...")
# Write the new data to a new CSV file
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['entryId', 'platform']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in new_data:
        writer.writerow(item)

print("Processing complete!")
