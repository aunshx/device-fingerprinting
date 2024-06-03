import json
import ast

def extract_and_concatenate_components(entry):
    entryId = entry[0]
    components_str = entry[2]
    try:
        components = ast.literal_eval(components_str)["fingerprint"]["components"]
        concatenated_string = []

        for component, details in components.items():
            value = details.get('value', "")
            duration = details.get('duration', 0)

            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            elif isinstance(value, list):
                value = ','.join(map(str, value))
            else:
                value = str(value)

            concatenated_string.append(value * duration)

        return {'entryId': entryId, 'concatenated_string': ''.join(concatenated_string)}
    except (SyntaxError, ValueError) as e:
        print(f"Error processing components for entryId {entryId}: {e}")
        return None
