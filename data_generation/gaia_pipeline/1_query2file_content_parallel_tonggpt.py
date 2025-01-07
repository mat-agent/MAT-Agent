# %%
import openai
from openai import AzureOpenAI
import json
import time
from tqdm import tqdm
import json 
import argparse
import os 
import sys
import string
import random
from concurrent.futures import ProcessPoolExecutor
from .merge import merge
def load_json(path):
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
system_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_system.prompt'
user_prompt_path = 'data_generation/gaia_pipeline/prompts/file/gaia_user.prompt'



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

def generate_identifier(length=16):
    characters = string.ascii_letters + string.digits + '_-'
    return ''.join(random.choices(characters, k=length))

def query2image_content_single_json(item, user_prompt_ori, system_prompt_ori,save_path):
    if "Query" in item.keys():
        query=item["Query"]
    elif "query" in item.keys():
        query=item["query"]
    else:
        return None
    
    if "Tools" in item.keys():
        tools=item["Tools"]
    elif "tools" in item.keys():
        tools=item["tools"]
    else:
        return None    

    if os.path.exists( save_path  + '/file/query2image_content_single/') is False:
        os.makedirs(save_path  + '/file/query2image_content_single/')
    save_path_new = save_path  + '/file/query2image_content_single/' + generate_identifier() + '.json'

    user_prompt=user_prompt_ori.replace('<query>', query) 
    user_prompt=user_prompt.replace('<suggested tools>', str(tools))

    dialogue = get_dialogue_system(user_prompt, system_prompt_ori, temperature=temperature, max_tokens=max_tokens)

    if '###' in dialogue:
        json_string=extract_between('### json start', '### json end', dialogue)
    elif '```' in dialogue:
        json_string=extract_between('```json\n', '```', dialogue)
    else:
        json_string=dialogue
    if json_string is not None:
        # count=count+1
        try:
            file_json = json.loads(json_string)
            file_json=file_json['file']
            item["files"]=file_json
            save_json(save_path_new, item)
        except:
            print('error in json saving')
    else:
        print('json is None')
    


def multi_procfess_query2image_content(args):
    with open(system_prompt_path, 'r') as file:
        system_prompt_ori = file.read()

    with open(user_prompt_path, 'r') as file:
        user_prompt_ori = file.read()
    source_query_path = args.save_path + '/query/queries_merged/'
    source_query_path = [file for file in os.listdir(source_query_path) if file.endswith('.json')][0]
    source_query_path = args.save_path + '/query/queries_merged/' + source_query_path
    if os.path.exists(args.save_path + '/file') is False:
        os.makedirs(args.save_path + '/file')
    save_path = args.save_path + '/file/query2image_content.json'


    data = load_json(source_query_path)

    start = time.time()
    with ProcessPoolExecutor(max_workers = args.workers ) as executor:
        for item in tqdm(data):
            #  print(item)
            executor.submit(query2image_content_single_json,item, user_prompt_ori,system_prompt_ori, args.save_path)
    
    print(f'Query to image content of {len(data)} samples cost time: {time.time() - start} s')




def main():

    parser = argparse.ArgumentParser(description='Generate queries using GAIA data')
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--save-path", type=str, default=f'data_generation/gaia_pipeline/save')
    
    args = parser.parse_args()
    timestamp=args.timestamp
    args.save_path=f'data_generation/gaia_pipeline/save/{timestamp}'

    if os.path.exists(f'data_generation/gaia_pipeline/log/{timestamp}') is False:
        os.mkdir(f'data_generation/gaia_pipeline/log/{timestamp}/')
    sys.stdout = open(f'data_generation/gaia_pipeline/log/{timestamp}/{timestamp}_1_q2content_parallel.log', 'a')

    multi_procfess_query2image_content(args)
    merge(args.save_path + '/file/query2image_content_single/', args.save_path + '/file/merged_json/', 'query2image_content.json')

main()