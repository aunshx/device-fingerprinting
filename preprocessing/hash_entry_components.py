from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import csv
from preprocessing.component_processor import extract_and_concatenate_components
from preprocessing.json_chunk_reader import read_json_in_chunks
from tqdm import tqdm

def extract_and_save_entry_components(input_filename, output_filename, hash_function, chunk_size=1000):
    num_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    print(f"Using {num_workers} threads to extract and concatenate components")

    # First pass to count total entries
    total_entries = 0
    for chunk in read_json_in_chunks(input_filename, chunk_size):
        total_entries += len(chunk)

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['entryId', 'hashed_string']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        completed_entries = 0
        print(f"Processing data from '{input_filename}'...")
        
        with tqdm(total=total_entries, desc="Processing entries", unit="entry") as pbar:
            for data_chunk in read_json_in_chunks(input_filename, chunk_size):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_entry, entry, hash_function) for entry in data_chunk]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            writer.writerow(result)
                
                completed_entries += len(data_chunk)
                pbar.update(len(data_chunk))

    print(f"Total entries processed: {completed_entries}")
    print(f"Results have been saved to '{output_filename}'")

def process_entry(entry, hash_function):
    result = extract_and_concatenate_components(entry)
    if result:
        concatenated_string = result['concatenated_string']
        hashed_string = hash_function(concatenated_string)
        return {'entryId': result['entryId'], 'hashed_string': hashed_string}
    return None