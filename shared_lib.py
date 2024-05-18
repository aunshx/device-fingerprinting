import json
import csv
import re

# function to parse the json file into a json object
def parse_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

# get finger print data from the json object
def get_finger_print_data(data, index = 2):
    return json.load(data[index])



