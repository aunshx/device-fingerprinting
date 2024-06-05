import json

# Load the original data from JSON file
with open('/Users/saishashetty/Desktop/device-fingerprinting/ml/data/new.json') as file:
    data = json.load(file)

# Process data
processed_data = []

for entry in data:
    platform = entry['platform']
    if platform in ['MacIntel', 'Win32']:
        processed_data.append(entry)
    elif platform in ['iPhone', 'iPad']:
        entry['platform'] = 'MacIntel'
        processed_data.append(entry)

# Save the processed data to a new JSON file
with open('ml/data/win_mac_test.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("Processed data saved as 'win_mac_test.json'")



