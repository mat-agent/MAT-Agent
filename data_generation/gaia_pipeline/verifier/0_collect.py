import json
import os
import argparse

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data


def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4) 



def list_files_in_folder(folder_path):
    try:
        # Get the list of files and directories in the specified folder
        files_and_dirs = os.listdir(folder_path)
        
        # Filter out directories, keeping only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
        
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
parser.add_argument("--timestamp", type=str)

args = parser.parse_args()
timestamp=args.timestamp

def list_files_in_directory(path):
    try:
        # Get a list of all files and directories in the given path
        items = os.listdir(path)
        
        # Filter out directories, keeping only files
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

json_list = list_files_in_directory(f'./data_generation/gaia_pipeline/save/{timestamp}/traj/')

# Example usage
json_root_path = './data_generation/gaia_pipeline/final_save/'

print ('json list', json_list)
save_name=f'all_json_{timestamp}_gpt4omini.json'


all_data=[]
for json_name in json_list:
    data = read_json(os.path.join(json_root_path,json_name))
    all_data=all_data+data

    save_json(os.path.join(json_root_path,save_name),all_data)

print ('total num', len(all_data))



