import asyncio
import os
from openai import OpenAI
import openai
from openai import AzureOpenAI
from typing import Optional
import json
import pandas as pd
import time
import base64
import json 
import argparse

from tongagent.llm_engine import QwenEngine
from tongagent.utils import load_config

# Set the relative path
relative_path = './'

# Change the working directory
os.chdir(relative_path)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data
    
def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4) 


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
system_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_file_verifier_system.prompt'
user_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_file_verifier_user.prompt'

file_save_root_path='./data/tongagent'



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



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
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0]:
                    return prediction
        except Exception as e:
            print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
            pass
    return ""

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
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response


def get_dialogue_system(content: str, system_prompt, temperature, max_tokens):

    try:
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
                    },
                ]
            }
        ]            
        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response






def get_dialogue_system_oneimage(content: str, system_prompt, images, temperature, max_tokens=2048):
    try:
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
        base64_images = []
        for image_path in images:
            base64_image = encode_image(image_path)
            base64_images.append(base64_image)  

        for base64_image in base64_images:
            messages[1]['content'].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            )
        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response


def get_dialogue_system_twoimage(content: str, system_prompt, image_path_1, image_path_2, temperature, max_tokens=2048):

    base64_image_1 = encode_image(image_path_1)
    base64_image_2 = encode_image(image_path_2)

    try:
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
                    },
                    {   
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}
                    },
                    {   
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image_2}"}
                    },
                ]
            }
        ]            
        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
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



from tongagent.tools.mdconvert import MarkdownConverter
md_converter = MarkdownConverter()

def get_dialogue_system_onefile(content: str, system_prompt, files, temperature, max_tokens=2048):
    try:
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
        file_results = []
        for file_path in files:
            if file_path[-4:] in ['.png', '.jpg']:
                base64_image = encode_image(file_path)
                file_results.append(base64_image)
            elif ".zip" in file_path:
                result = md_converter.convert(file_path)
                file_results.append(result.text_content)
            else:
                result = md_converter.convert(file_path)
                file_results.append(result)  

        count=0
        for file_result in file_results:
            file_path = files[count]

            if file_path[-4:] in ['.png', '.jpg']:
                messages.append(
                    {
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{file_result}"}
                            }                 
                        ]
                    }
                )
            elif ".zip" in file_path:
                messages.append(
                    {
                        "role": "user",
                        "content": "Here is a zip file:\n### "
                        + str(file_result)

                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Here is a file:\n### "
                        + str(file_result.title)
                        + "\n\n"
                        + file_result.text_content[:70000],
                    },
                )
            count=count+1

        response = get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass

    return response



from IPython.display import Image, display, HTML
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
    parser.add_argument("--timestamp", type=str)

    args = parser.parse_args()
    timestamp=args.timestamp

    save_path='./data_generation/gaia_pipeline/final_save/'

    q_description_filepath_path = save_path + f'all_json_{timestamp}_gpt4omini.json'
    q_description_filepath_filter_path = save_path + f'/all_json_{timestamp}_gpt4omini_queryfile.json'
    q_description_filepath_filter_COT_path = save_path + f'/all_json_{timestamp}_gpt4omini_queryfile_COT.json'




    os.chdir(os.path.dirname(relative_path))
    with open(system_prompt_path, 'r') as file:
        system_prompt_ori = file.read()

    with open(user_prompt_path, 'r') as file:
        user_prompt_ori = file.read()

    def fetch_content(item):
        if "file_information" not in item:
            return []
        file = item["file_information"]
        content = []
        for i in range(len(file)):
            j = file[f'file_{str(i+1)}']
        # for i,j in file:
            content.append(j['file_content'])
        return content


    data = read_json(q_description_filepath_path)
    new_data = []
    filter_cot = []
    for idx, item in tqdm(enumerate(data)):
        # if idx > 10:
        #     break
        if 'correct' in item.keys():
                if item['correct'] == 'yes':
                    new_data.append(item)
                    continue
                else:
                    query = item['updated_query']
        else:
            query = item['query']

        print("-" * 50)
        print(f"ID: {idx}. Query: {item['query']}")

        file_list=item['file']['file_name']


        for i in range (len(file_list)):
            file_list[i]=os.path.join(file_save_root_path,file_list[i])

        user_prompt=user_prompt_ori.replace('<query>', query)

        print("-----------------" * 3)
        print(f"System prompt: {system_prompt_ori}")
        print(f"User prompt: {user_prompt}")
        if len(file_list) == 0:
            image_content='no image provided'
            user_prompt=user_prompt.replace('<image_content>', image_content)
        else:
            pass
        
        dialogue = get_dialogue_system_onefile(user_prompt, system_prompt_ori, file_list, temperature=temperature, max_tokens=max_tokens)
        print(f"Dialogue: {dialogue}")
        try:
            dialogue_json = dialogue.split('```json')[1].split('```')[0]
            dialogue_json = json.loads(dialogue_json) 
        except:
            try:
                dialogue_json = dialogue.split('### start json')[1].split('### end json')[0]
                dialogue_json = json.loads(dialogue_json) 
            except:
                dialogue_json = dialogue
                print(f"Dialogue format error, failure in json save: {dialogue}")
                continue
        
        filter_cot.append(dialogue_json)

    
        correctness = dialogue_json['correct']
        updated_query = dialogue_json['updated_query']
        item['correct'] = correctness
        item['updated_query'] = updated_query
        new_data.append(item)

        # print(f"Dialogue: {dialogue}")
        print("-" * 50)
    save_json(q_description_filepath_filter_path, new_data)
    save_json(q_description_filepath_filter_COT_path, filter_cot)



    # write a code to count the number of correct and incorrect
    from collections import Counter
    correctness = [item['correct'] for item in new_data]
    counter = Counter(correctness)
    print(f"Count of yes : {counter['yes']}")
    print(f"Count of no : {counter['no']}")

main()
