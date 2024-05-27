import json
import csv
import re

def get_os_from_user_agent(user_agent):
    # pull all the data from inside the first set of ()
    match = re.search(r"\((.*?)\)", user_agent)
    return match.group(1) if match else ""
    
def extract_user_agent(header):
    # pull the user agent from the header
    match = re.search(r"User-Agent: (.*?)(?:\n|$)", header)
    return match.group(1) if match else ""

def process_json(json_file, csv_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Open CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Entry ID', 'User-Agent OS'])

        # Loop through the JSON array
        for entry in data:
            entry_id = entry[0]
            headers = entry[1]
            user_agent = extract_user_agent(headers)
            user_os = get_os_from_user_agent(user_agent)
            csvwriter.writerow([entry_id, user_os])

if __name__ == "__main__":
    process_json('./data/HASHED_combined_data_upto_oct_31_2023.json', './data/user-agent-os.csv')
