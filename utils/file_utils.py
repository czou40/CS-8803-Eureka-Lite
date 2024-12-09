import os
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import glob

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def find_screenshots(iteration:int, response_id:int):
    # Use glob to find matching files in the current working directory
    search_pattern = f'iter{iteration}_response{response_id}_*.jpg'
    files = glob.glob(search_pattern)

    # Sort the file paths in ascending order
    sorted_files = sorted(files)

    return sorted_files    

# def load_tensorboard_logs(path):
#     data = defaultdict(list)
#     event_acc = EventAccumulator(path)
#     event_acc.Reload()  # Load all data written so far

#     for tag in event_acc.Tags()["scalars"]:
#         events = event_acc.Scalars(tag)
#         for event in events:
#             data[tag].append(event.value)
    
#     return data

def load_logs(path):
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            tag, value = line.strip().split(',')
            data[tag].append(float(value))
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function

if __name__ == '__main__':
    # Test the function
    # files = find_files_with_substring(os.getcwd(), 'iter0_response0')
    # print(files)
    # print(find_screenshots(0, 0))
    # print(load_tensorboard_logs('logs/2021-10-06_14-57-45'))
    cwd = '/Users/chunhaozou/Desktop/drl/outputs/eureka/2024-12-08_18-10-39'
    # change current working directory to the path of the logs
    os.chdir(cwd)
    print(find_screenshots(0, 0))
