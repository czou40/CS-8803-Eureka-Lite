import json

def pretty_print_yellow(data):
    if not isinstance(data, str):
        data = json.dumps(data, indent=4, sort_keys=True)
        
    # Print the JSON data with yellow color
    print("\033[93m" + data + "\033[0m")
