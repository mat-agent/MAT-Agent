import matplotlib
matplotlib.use('Agg')

import os
from openai import OpenAI
from typing import Optional
import json
import pandas as pd
import sys
import argparse
import json 
from huggingface_hub import login
from tqdm import tqdm
from openai import AzureOpenAI
import os
from tongagent.agents.data_sampling_agent import create_agent
react_agent = create_agent(task='gaia')

file_save_root_path='data/tongagent'
 
import sys
from datetime import datetime

class CircularReferenceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.default(i) for i in obj]
        else:
            return str(obj)  # or handle other types as needed
        
 

# Open the log file

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data
    
def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
 

def traj_generation(args):

    if os.path.exists(args.save_path + '/traj') is False:
        os.makedirs(args.save_path + '/traj')   
        
    data_traj_save_path= args.save_path + f'/traj/gaia_traj_{args.start}_{args.end}.json'
    json_root_path = f"{args.save_path}/file/merged_json/query2image_content_filepath_{args.start}_{args.end}.json"
    with open(os.path.join(json_root_path), 'r') as f:
        data = json.load(f)
 
    new_data=[]
    count = 0
    for i in tqdm(data):

        query = i['query']

        if 'files' not in i:
            print('No files in extracted_data')
            question = f"{query}\n"

        else:
            files = i['files']
            images = []

            for path in i["file_name"]:
                images.append(os.path.join(file_save_root_path, path))

            attachment = ""
            for image in images:
                attachment=attachment + image + "; "

            if len (images)>0:
                question = f"{query}\n Attachment: {attachment} "
            else:
                question = f"{query}\n"
            # react_agent.set_image_paths(images)
        print ('================question',question)
        res = react_agent.run(question)
        traj_js=react_agent.write_inner_memory_from_logs()
        print("traj-----------------------------------------------------")
        for tj in traj_js:
           print(f'\titem: {tj}')
        i['traj']=traj_js
        i['answer']=str(res)
        new_data.append(i)
        count += 1

        try:
            save_json(data_traj_save_path,new_data)
        except:
            print(f'Failed to save traj {count} to json file')


def main():
    parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--save-path", type=str, default=f'data_generation/gaia_pipeline/save')
    args = parser.parse_args()
    timestamp=args.timestamp
    args.save_path=f'data_generation/gaia_pipeline/save/{timestamp}'

    traj_generation(args)

main()



