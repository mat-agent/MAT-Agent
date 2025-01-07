import json
import os
import argparse

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f) 
def save_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)    


parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
parser.add_argument("--timestamp", type=str)

args = parser.parse_args()
timestamp=args.timestamp


former = f'./data_generation/gaia_pipeline/final_save/all_json_{timestamp}_gpt4omini_queryfile.json'
data_former = load_json(former) 
print(len(data_former))
minicpm_save_path = f'./data_generation/gaia_pipeline/final_save/all_json_{timestamp}_gpt4omini_minicpm.json'




import random
import string
def random_id(length):
    return ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=length))

import json

def filter_wired(data):

    saved = []
    for item in data:
        
        conv = item["conversations"]
        
        followed_instruction = []
        has_error = []
        for turn_id, turn in enumerate(conv):
            if turn["role"] == "user": 
                if turn_id > 1:
                    has_error.append("Error in code parsing:" in turn["content"])
                continue
            
            followed_instruction.append('Thought:' in turn["content"] and 'Code:' in turn["content"])
        
        
        if all(has_error) == True:
            continue
            
        if not any(followed_instruction):
            if not all(has_error):
                saved.append(item)
            
        else:
            saved.append(item)

        continue
        
    # break
    
    return saved




from collections import Counter


new_data  =[]
file_type = [".xml", ". html", ".mp3", ".xlsx", ".txt", ".pdf", ".json", ".csv", ".docx", ".py", ".png", ".jpg","image"]
image_type = [".png", ".jpg", "image"]   
types = []
types_filtered = []
for i in data_former:
    traj_former = i['traj']
    tmp = {

        "id": random_id(8),
        "image":{},
        "system-prompt":traj_former[0]['content'],
        # "correct": i['correct'],
        "answer": i['answer']  

    }
    
    
    traj_new = [
        {   "role":"user",
            "content":f"{traj_former[1]['content']}"
        }
    ]
    if 'files' in i:
        mark = 1
        for file in i['files']:
            types.append(file['type'])

        if len(i['files']) ==   1:  
            
            if i['files'][0]['type'] not in file_type:
                continue
            else:
                types_filtered.append(i['files'][0]['type'])
            if i['files'][0]['type']  in image_type:
                print(i['files'][0]['path'])
                tmp['image'] = i['files'][0]['path']
                traj_new[0]['content'] = f'<image>\n' + traj_new[0]['content']
        else:
            file_image = [fi['path'] for fi in i['files'] if fi['type'] == 'image']
            if len(file_image) >  len(list(set(file_image))):
                continue
            if len(file_image) > 3:
                continue
            tmp_image = {  }
            for j in range(len(i['files'])-1, -1, -1):
                
                if i['files'][j]['type'] not in file_type:
                    mark = 0
                else:
                    types_filtered.append(i['files'][j]['type'])
                if i['files'][j]['type']  in image_type:
                    print(i['files'][j]['path'])
                    tmp_image = {f'<image_0{j}>': i['files'][j]['path']}
                    tmp['image'].update(tmp_image)
                    traj_new[0]['content'] = f'<image_0{j}>\n' + traj_new[0]['content']
            if mark == 0:
                continue
    else:
        tmp['image'] = {}


    for t in range(2, len(traj_former)-1):

        if traj_former[t]['role'] == 'user' or traj_former[t]['role'] == 'tool-response':
            traj_new.append(
                {   "role":"user",
                    "content": traj_former[t]['content']
                }
            )
        elif traj_former[t]['role'] == 'assistant':
            traj_new.append(
                {   "role":"assistant",
                    "content": traj_former[t]['content']
                }
            )
    tmp['conversations'] = traj_new
    new_data.append(tmp)
    

new_data = filter_wired(new_data)   
# new_data = new_data + level3
print(f"New data type {len(new_data)}")

counter = Counter(types)
print(counter)
counter = Counter(types_filtered)   
print(counter)

random.shuffle(new_data)


save_json(minicpm_save_path, new_data)


import json
from tqdm import tqdm
import re

# Regular expression pattern to match the desired format
pattern = r'<image_(\d{2})>'

def debugger(minicpm_save_path):
    with open(minicpm_save_path, "r") as f:
        dataset = json.load(f)
    count = 0 
    for item_id, item in tqdm(enumerate(dataset)):
        if 'image' not in item:
            print(item_id)
            print(json.dumps(item, indent=4, ensure_ascii=False))
            print(json.dumps(dataset[item_id-1], indent=4, ensure_ascii=False))
            
            exit()
        image = item['image']
        prompt = item['conversations'][0]["content"]
        if type(image) is str and len(image) > 0:
            assert "<image>" in prompt
        else:
            # print(len(item['image']))
            prompt = item['conversations'][0]["content"]
            n_images = len(image)
            for k, v in image.items():
                if k in prompt:
                    continue
                print(prompt)
                print(json.dumps(item, indent=4, ensure_ascii=False))
                count += 1        
            # Find all matches
            matches = re.findall(pattern, prompt)

            # Print the matches
            # print(matches)
            if len(matches) > 0:
                for match in matches:
                    to_check = f"<image_{match}>"
                    if to_check not in image:
                        print(prompt)
                        print(image)
                        print(json.dumps(item, indent=4, ensure_ascii=False))
                        count += 1
    print("Problematic", count)
    assert count == 0
debugger(minicpm_save_path) 
