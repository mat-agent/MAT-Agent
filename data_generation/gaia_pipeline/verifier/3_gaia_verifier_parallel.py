# %%
import asyncio
import os
from openai import OpenAI
import openai
from openai import AzureOpenAI
from typing import Optional
import json
# import pandas as pd
import time
import string
import random
import base64
import json 
import argparse
from tqdm import tqdm
import argparse

# from collections import Counter
# from concurrent.futures import ProcessPoolExecutor
 
def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data
    
def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4) 



parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
parser.add_argument("--timestamp", type=str)
parser.add_argument("--number", type=int, default=-1)

args = parser.parse_args()
timestamp=args.timestamp


from tongagent.utils import load_config
config = load_config()

max_tokens = 2048
temperature = 1

if config.data_generation.llm=='azure':
    ### use Azure API to call chatgpt
    REGION = config.data_generation.region
    MODEL = config.data_generation.model
    API_KEY = config.data_generation.api_key
    API_BASE = config.data_generation.ape_base
    ENDPOINT = f"{API_BASE}/{REGION}"
    NUM_SECONDS_TO_SLEEP=10
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint=ENDPOINT,
    )
elif config.data_generation.llm=='openai':
    ### use OpenAI API to call chatgpt
    import openai
    MODEL = config.data_generation.model
    openai.api_key = os.environ["OPENAI_API_KEY"]
    from openai import OpenAI
    client = OpenAI()


### prompt setting
system_prompt_path = './data_generation/file_generation/prompt/gaia_verifier_system.prompt'
user_prompt_path = './data_generation/file_generation/prompt/gaia_verifier_user.prompt'
tool_description_path='./data_generation/file_generation/prompt/tool_description.json'

q_file_trajectory_path=f'./data_generation/gaia_pipeline/final_save/all_json_{timestamp}_gpt4omini_minicpm.json'
file_filtered_folder=f'./data_generation/gaia_pipeline/final_save/before_{timestamp}_gpt4omini/'
final_save_folder='./data_generation/gaia_pipeline/final_save/'
final_save_name=f'all_json_{timestamp}_gpt4omini_minicpm_train.json'
# %%




def encode_image(image_path):
    # print("image_path", image_path) 
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        # print(e)
        return None 



def get_chat_response(messages, model=MODEL, temperature=1, max_tokens=2048, n=1, patience=10, sleep_time=2):
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            print("response", response) 
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0]:
                    return prediction
        except Exception as e:
            # print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
            pass
    return "nonononon"

def get_dialogue(content: str, max_tokens):

    try:
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        "type": "text",
                        "text": content
                    },
                ]
            }
        ]            
        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        # print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        # print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response


def get_dialogue_system_oneimage(content: str, system_prompt, images, temperature, max_tokens=2048):

    # print("=====1=====")
    try:
        # print("=====1=====")
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },            
            {
                'role': 'user',
                'content': [
                    {
                        "type": "text",
                        "text": content
                    }                 
                ]
            }
        ]    
        # print("=====images=====", images)      
        base64_images = []
        for image_path in images:
            base64_image = encode_image(image_path)
            # print("=====base64_image====q=")
            base64_images.append(base64_image)  
        # print("=====images1=====", images)      
        for base64_image in base64_images:
            messages[1]['content'].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            )
        # print("=====1=====")
        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
        # print("=============")
    except openai.error.RateLimitError:
        # print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        # print(e)
        # print('Error in get_dialogue_system_oneimage')
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response


def extract_between(str1, str2, str3):
    try:
        # Find the starting index of str1 in str3
        start_index = str3.find(str1)
        if start_index == -1:
            return None  # str1 not found in str3
        
        # Adjust the start_index to the end of str1
        start_index += len(str1)
        
        # Find the ending index of str2 in str3, starting from start_index
        end_index = str3.find(str2, start_index)
        if end_index == -1:
            return None  # str2 not found in str3
        
        # Extract the substring between str1 and str2
        return str3[start_index:end_index]
    except Exception as e:
        return None

def traj_extraction(traj_json):

    trajectory=''
    step_num=len(traj_json)
    count=1
    for step in traj_json:
        if step['role'] =='assistant':
            trajectory_step=step['content']         
            trajectory=trajectory + trajectory_step + '\n'

        elif step['role'] == 'user': 
            trajectory_step=step['content']
            trajectory=trajectory + trajectory_step + '\n'

    return trajectory

def generate_identifier(length=16):
    characters = string.ascii_letters + string.digits + '_-'
    return ''.join(random.choices(characters, k=length))


def traj_verifier_single(task,user_prompt_ori,system_prompt_ori,tool_set,file_filtered_folder):
    save_path = file_filtered_folder + '/' + generate_identifier() + '.json'

    conversation = task['conversations']
    query = conversation[0]['content']
    # # print ('=============query================',query)
    user_prompt=user_prompt_ori.replace('<query>', query)
    user_prompt=user_prompt.replace('<tool description>', "TOOL_SET")

    traj_json=conversation[1:]
    traj_string=str(traj_json)
    # # print (traj_string)
    user_prompt=user_prompt.replace('<traj>', traj_string)

    image_files=task['image']
    # print(f"==============")

    if 'dict' in str(type(image_files)):
        image_files=list(image_files.values())
    elif 'str' in str(type(image_files)):
        image_files=[image_files]

    # print(f"==============")
    image_num=len(image_files)

    prediction=task['answer']
    user_prompt=user_prompt.replace('<execution_result>', str(prediction))


    image_paths=[]
    dialogue = get_dialogue_system_oneimage(user_prompt, system_prompt_ori, image_paths, temperature=0.2, max_tokens=2048 *2)


    if '"correct": "no"' in dialogue:
        correctness = 'no'
    elif '"correct": "yes"' in dialogue:
        correctness = 'yes'
    else:
        correctness = 'unknown'  
    task["traj_verification"] = dialogue
    task['correct'] = correctness
 
    # print(save_path)
    save_json(save_path, task)  
    return None
 
def traj_verifier(args):
    with open(system_prompt_path, 'r') as file:
        system_prompt_ori = file.read()

    with open(user_prompt_path, 'r') as file:
        user_prompt_ori = file.read()

    with open(tool_description_path, 'r') as file:
        tool_set = file.read()

    data = read_json(q_file_trajectory_path)
    if args.number > 0:
        data = data[:args.number]   
    else:
        data = data
    # print(f"Traj verifier of {len(data)} samples")
    # for task in data:
    start = time.time()
    print('start time', time.time())
    # with ProcessPoolExecutor(max_workers = args.workers ) as executor:
    #     for item in tqdm(data):
    #         #  # print(item)
    #         executor.submit(traj_verifier_single, item, user_prompt_ori, system_prompt_ori,tool_set, file_filtered_folder)
    
    for task in tqdm(data):
        traj_verifier_single(task, user_prompt_ori, system_prompt_ori, tool_set, file_filtered_folder)
        print ('current time:',time.time()  )
    print(f'Traj verifier of {len(data)} samples cost time: {time.time() - start} s')

    # print(f'Traj verifier of {len(data)} samples cost time: {time.time() - start} s')
 
def merge(source_folder , output_folder, filename):
    # source_folder = path + '/query/query_json/'
    # output_folder = path + '/query/queries_merged'
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)  

    save_path = os.path.join(output_folder, filename)      
    json_files = [pos_json for pos_json in os.listdir(source_folder) if pos_json.endswith('.json')]
    data = []
    for json_file in json_files:
        with open(os.path.join(source_folder, json_file)) as f:
            tmp = json.load(f)
            if isinstance(tmp, list) and len(tmp) == 1:
                tmp = tmp[0]
            if isinstance(tmp, list):
                data += tmp
            else:
                data.append(tmp)
    length = len(data)

    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder)

    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"Successfully merged {length} json files")



traj_verifier(args)
merge(file_filtered_folder , final_save_folder, final_save_name)


 


