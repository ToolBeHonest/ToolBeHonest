import json

def load_results(results_file):
    with open(results_file, "r") as f:
        return json.load(f)

def save_results(file_path, data):
    with open(file_path, "w") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)

def save_table(file_path, data):
    with open(file_path, "w") as fw:
        fw.write(data)