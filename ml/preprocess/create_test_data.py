import json

# Load the JSON file
file_path = '../data_current/extracted_component_data.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the required data into the desired format
result = []
for entry in data:
    try:
        operation_details = json.loads(entry["operationDetails"].replace("'", "\""))
        if operation_details.get("executableName") == "exec_step":
            result.append({
                "entryId": entry["entryId"],
                "platform": entry["components"]["platform"]["value"],
                "operationOutput": entry["operationOutput"]
            })
    except (KeyError, json.JSONDecodeError, TypeError) as e:
        print(f"Skipping entry due to error: {e}")

# Save the result to a JSON file
output_file_path = '../data_current/processed_data.json'
with open(output_file_path, 'w') as output_file:
    json.dump(result, output_file, indent=4)

print(f"Processed data saved to {output_file_path}")

# Load the JSON file
file_path = '../data_current/processed_data.json'
with open(file_path, 'r') as file:
    data2 = json.load(file)

def count_platforms(data):
    platform_counts = {}

    for entry in data:
        platform = entry['platform']
        if platform in platform_counts:
            platform_counts[platform] += 1
        else:
            platform_counts[platform] = 1
    
    print(platform_counts)

    return platform_counts

count_platforms(data2)
