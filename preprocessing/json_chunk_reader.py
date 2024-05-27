import json

def read_json_in_chunks(file, chunk_size=1000):
    """Read a large JSON file in chunks."""
    with open(file, 'r') as f:
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
