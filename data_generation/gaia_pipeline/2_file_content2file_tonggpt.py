import asyncio
import os
from openai import OpenAI
import openai
from openai import AzureOpenAI
from typing import Optional
import json
import pandas as pd
import time
import numpy as np
import json 
import shutil

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




def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data
    
def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4) 



### prompt setting
system_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_file_generation_system.prompt'
user_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_file_generation_user.prompt'
file_save_root_path = 'data/tongagent'

query_embedding_save_path = config.data_generation.query_embedding_save_path
image_base_path = config.data_generation.image_base_path
caption_data_path = config.data_generation.caption_data_path


# query_embedding_save_path='/home/zhigao/scratch2/TongAgent/data_generation/file_generation/source/support_embedding_sharegpt4v_100k_chartqa_all.json'
# image_base_path='/media/disk3/meta_data/open_llava_next'
# caption_data_path = '/home/zhigao/scratch2/TongAgent/data_generation/file_generation/source/chartqa_sharegpt4v_all.json'
# file_save_root_path='data/tongagent'


import random
import string

def generate_random_string(length=16):
    # characters = string.ascii_letters + string.digits + string.punctuation
    characters = string.ascii_letters
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# # Generate a random string with 16 characters
# random_string = generate_random_string()

def extract_file_name(file_path):
    return os.path.basename(file_path)

def get_folder_path(file_path):
    folder_path = os.path.dirname(file_path)
    return folder_path




def get_chat_response(messages, model=MODEL, temperature=1, max_tokens=2048, n=1, patience=10, sleep_time=2):
    while patience > 0:
        patience -= 1
        try:
            # print(" CONNECTING TO GPT4-V")
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


def extract_between(str1, str2, str3):
    try:
        # Find the starting index of str1 in str3
        start_index = str3.find(str1)
        print ('start_index',start_index)
        if start_index == -1:
            return None  # str1 not found in str3
        
        # Adjust the start_index to the end of str1
        start_index += len(str1)
        
        # Find the ending index of str2 in str3, starting from start_index
        end_index = str3.find(str2, start_index)
        print ('end_index',end_index)
        if end_index == -1:
            return None  # str2 not found in str3
        
        # Extract the substring between str1 and str2
        return str3[start_index:end_index]
    except Exception as e:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()


    # with open('data_generation/gaia_pipeline/_timestamp.txt', 'r') as f:
    #     timestamp = f.read().strip()

    timestamp=args.timestamp

    save_path=f'data_generation/gaia_pipeline/save/{timestamp}'
    q_description_path = save_path + '/file/merged_json/query2image_content.json'
    q_description_filepath_path = save_path + f'/file/merged_json/query2image_content_filepath_{args.start}_{args.end}.json'


    if not os.path.exists(os.path.join(file_save_root_path,'multimodal_file')):
        os.makedirs(os.path.join(file_save_root_path,'multimodal_file'))

    from FlagEmbedding import BGEM3FlagModel
    BGE_model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    caption_image_data = read_json(caption_data_path)
    # data_num=len(caption_image_data)
    # caption_list=[]
    # for i in range(data_num):
    #     data_i= caption_image_data[i]
    #     caption=caption_image_data[i]['conversations'][1]['value']
    #     caption_list.append(caption)
    # support_embedding = BGE_model.encode(caption_list)['dense_vecs']
    
    support_embedding = np.load(query_embedding_save_path)


    # from IPython.display import Image, display, HTML


    with open(system_prompt_path, 'r') as file:
        system_prompt_ori = file.read()

    with open(user_prompt_path, 'r') as file:
        user_prompt_ori = file.read()


    q_description_json = read_json(q_description_path)
    q_description_imagepath_json=[]

    count=0
    task_num=0      
    for task in q_description_json:

        task_num=task_num+1
        if task_num>args.start and task_num<=args.end:

            # if count>1000:
            #     break
            print ('=======================================================================================')
            print ('===========count=============count==================count=====================count========================',count)
            if "Query" in task.keys():
                query=task["Query"]
            elif "query" in task.keys():
                query=task["query"]
            else:
                continue
            print ('=============query================',query)
            file_json=task["files"]
            file_num=file_json["file_numbers"]
            file_information=file_json["file_information"]
            query_file_num=len(file_information)

            all_path=[]
            file_list=[]

            flag=0
            for i in range (query_file_num):
                file_list_one={}
                one_file=file_information["file_"+str(i+1)]   #file_1
                file_type=one_file["file_type"]
                file_content=one_file["file_content"]
                print ('file_content',file_content)
                
                if 'jpg' in file_type or 'png' in file_type:

                    # query_image_embedding= BGE_model.encode([file_content])['dense_vecs']

                    try:
                        query_image_embedding= BGE_model.encode([file_content])['dense_vecs']
                    except:
                        print('error in query embedding extraction')
                        continue

                    print ('query_image_embedding',query_image_embedding.shape)
                    print ('support_embedding',support_embedding.shape)
                    similarity = query_image_embedding @ support_embedding.T
                    print ('similarity',similarity.shape)

                    # max_similarity = np.max(similarity, axis=1)
                    max_indices = np.argmax(similarity, axis=1)

                    file_path=caption_image_data[max_indices[0]]["image"]
                    # one_file['file_path']=file_path
                    file_name=extract_file_name(file_path)
                    file_folder=get_folder_path(file_path)

                    try:
                        if os.path.exists(os.path.join(file_save_root_path,file_folder)) is False:
                            os.makedirs(os.path.join(file_save_root_path,file_folder))

                        if os.path.exists(os.path.join(file_save_root_path,file_path)):
                            print ('image exists')
                        else:
                            shutil.copy(os.path.join(image_base_path,file_path), os.path.join(file_save_root_path,file_folder))
                            print ('successful image copy')
                    except:
                        flag=1
                        print ('image copy fails')
                        break

                    file_list_one["type"]="image"
                    file_list_one["path"]=file_path

                else:
                    file_name=generate_random_string()+file_type
                    file_path='multimodal_file/'+file_name
                    system_prompt=system_prompt_ori.replace("<file type placeholder>", file_type)
                    user_prompt=user_prompt_ori.replace("<file type placeholder>", file_type)
                    user_prompt=user_prompt.replace("<file content>", file_content)
                    user_prompt=user_prompt.replace("<file name>", file_name)
                    user_prompt=user_prompt.replace("<save path>", file_save_root_path+'/multimodal_file')

                    dialogue = get_dialogue_system(user_prompt, system_prompt_ori, temperature=temperature, max_tokens=max_tokens)
                    # dialogue = qwen_llm.forward(system_prompt_ori, user_prompt)
                    print ('dialogue',dialogue)
                    code=extract_between("##code start", "##code end", dialogue)

                    # if code[:10]=="\n```python" and code [:-4]=="```\n":
                    #     code=code[10:-4]

                    # if "```python" in code and "```" in code:
                    #     code=extract_between("```python", "```", code)

                    if code ==None:
                        flag=1
                        break
                    else:
                        code=code.replace("’","'")
                        code=code.replace("you’","you'")
                        code=code.replace("It’","It'")
                        code=code.replace("I’","I'")
                        code=code.replace("’s","'s")
                        for prime in range (10):
                            code=code.replace("’","'")
                            code=code.replace("you’","you'")
                            code=code.replace("It’","It'")
                            code=code.replace("I’","I'")
                            code=code.replace("’s","'s")                              
                        print ('=============code start================')
                        print (code)
                        print ('=============code end================')
                        code.replace("’", "")

                    try:
                        exec(code)
                        print ('code success')
                    except:
                        flag=1
                        print ('code fails')
                        break
                    # one_file['file_path']=file_name

                    file_list_one["type"]=file_type
                    file_list_one["path"]=file_path

                file_list.append(file_list_one)
                all_path.append(file_path)

            if flag==0:
                count=count+1
                task["file_num"]=query_file_num
                task["files"]=file_list
                task["file_name"]=all_path
                q_description_imagepath_json.append(task)
                  
        save_json(q_description_filepath_path, q_description_imagepath_json)
    save_json(q_description_filepath_path, q_description_imagepath_json) 


main()