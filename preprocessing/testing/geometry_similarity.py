import json

def extract_geometry(data):
    geometry_data = {}
    for device_data in data:
        try:
            # Parse the JSON string
            fingerprint_str = device_data[2].replace("'", "\"")
            fingerprint = json.loads(fingerprint_str)
            
            # Extract the fingerprintID and geometry value
            fingerprint_id = fingerprint.get('fingerprint', {}).get('visitorId', '')
            canvas = fingerprint.get('fingerprint', {}).get('components', {}).get('canvas', {})
            geometry = canvas.get('value', {}).get('geometry', '')
            
            # Store the geometry value keyed by fingerprintID
            if fingerprint_id and geometry:
                geometry_data[fingerprint_id] = geometry
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error for device_data: {device_data[:100]}...")  # Log the error and the first 100 characters
            print(f"Error message: {str(e)}")
    return geometry_data

# Load the data from a single file for demonstration
data = json.load(open('/mnt/data/chunk_54.json'))
geometry_data = extract_geometry(data)

# Print the extracted geometry data
for fingerprint_id, geometry in geometry_data.items():
    print(f"FingerprintID: {fingerprint_id}, Geometry: {geometry[:100]}...")  # Print the first 100 characters of geometry for brevity
