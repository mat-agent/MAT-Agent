import argparse
import json
import os
import time
import openai
from openai import AzureOpenAI
from tqdm import tqdm 
import requests
from concurrent.futures import ProcessPoolExecutor
import time
# from .gta_query_generation import *
import json 
import random
import argparse
import string
import sys

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



# timestamp = time.strftime("%Y%m%d-%H%M%S")
# with open('data_generation/gaia_pipeline/_timestamp.txt', 'w') as f:
#     f.write(timestamp)


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)




def remove_between_characters(s, a, b):
    # Find the first occurrence of characters a and b
    start = s.find(a)
    end = s.find(b, start + 1)
    # If either character is not found, return the original string
    if start == -1 or end == -1:
        return s
    # Create a new string by excluding the part between a and b
    new_string = s[:start] + s[end + 1:]
    return new_string


json_prompt_paths = [
   {
        "json": "data_generation/gaia_pipeline/prompts/query/gaia_val_metadata.jsonl",
        "prompt": "data_generation/gaia_pipeline/prompts/query/gaia_val_query_generation.prompt"
   }
]

def merge(path , timestamp):
    json_path = path + '/query/query_json/'
    merge_save_path = path + '/query/queries_merged'
    json_files = [pos_json for pos_json in os.listdir(json_path) if pos_json.endswith('.json')]
    data = []
    for json_file in json_files:
        with open(os.path.join(json_path, json_file)) as f:
            tmp = json.load(f)
            if isinstance(tmp, list) and len(tmp) == 1:
                tmp = tmp[0]
            data += tmp
    length = len(data)

    if os.path.exists(merge_save_path):
        pass
    else:
        os.makedirs(merge_save_path)

    with open(f"{merge_save_path}/gaia_query_num{length}_{timestamp}.json", 'w') as f:
        json.dump(data, f)
    print(f"Successfully merged {length} json files")


tool_map = {
    'web browser': 'ask_search_agent', 
    'search engine': 'ask_search_agent', 
    'calculator': 'PythonInterpreter', 
    'image recognition tools': 'visualizer', 
    'none': 'PythonInterpreter', 
    'a web browser': 'ask_search_agent', 
    'a search engine': 'ask_search_agent', 
    'pdf access': 'inspect_file_as_text', 
    'pdf viewer': 'inspect_file_as_text', 
    'microsoft excel': 'inspect_file_as_text', 
    'image recognition': 'visualizer', 
    'a calculator': 'PythonInterpreter', 
    'ocr': 'visualizer', 
    'python': 'PythonInterpreter', 
    'video recognition tools': 'visualizer', 
    'microsoft excel / google sheets': 'inspect_file_as_text', 
    'excel': 'inspect_file_as_text', 
    'color recognition': 'visualizer', 
    'excel file access': 'inspect_file_as_text', 
    'access to wikipedia': 'ask_search_agent', 
    'image recognition/ocr': 'visualizer', 
    'a file interface': 'inspect_file_as_text', 
    'a web browser.': 'ask_search_agent', 
    'a search engine.': 'ask_search_agent', 
    'file handling': 'inspect_file_as_text', 
    'a speech-to-text tool': 'inspect_file_as_text', 
    'audio capability': 'inspect_file_as_text', 
    'unlambda compiler': 'inspect_file_as_text', 
    'a calculator.': 'PythonInterpreter', 
    'google search': 'ask_search_agent', 
    'jsonld file access': 'inspect_file_as_text', 
    'video parsing': 'visualizer', 
    'python compiler': 'PythonInterpreter', 
    'word document access': 'inspect_file_as_text', 
    'tool to extract text from images': 'visualizer', 
    'a word reversal tool / script': 'inspect_file_as_text', 
    'counter': 'PythonInterpreter', 
    'xml file access': 'inspect_file_as_text', 
    'access to the internet archive, web.archive.org': 'ask_search_agent', 
    'text processing/diff tool': 'inspect_file_as_text', 
    'gif parsing tools': 'visualizer', 
    'code/data analysis tools': 'inspect_file_as_text', 
    'pdf reader': 'inspect_file_as_text', 
    'markdown': 'inspect_file_as_text', 
    'google translate access': 'PythonInterpreter', 
    'bass note data': 'inspect_file_as_text', 
    'text editor': 'inspect_file_as_text', 
    'xlsx file access': 'inspect_file_as_text', 
    'powerpoint viewer': 'inspect_file_as_text', 
    'csv file access': 'inspect_file_as_text', 
    'computer algebra system': 'inspect_file_as_text', 
    'video processing software': 'visualizer', 
    'audio processing software': 'inspect_file_as_text', 
    'computer vision': 'visualizer', 
    'google maps': 'ask_search_agent', 
    'access to excel files': 'inspect_file_as_text', 
    'a python ide': 'inspect_file_as_text', 
    'spreadsheet editor': 'inspect_file_as_text', 
    'no tools required': 'PythonInterpreter', 
    'image recognition and processing tools': 'visualizer', 
    'computer vision or ocr': 'visualizer', 
    'c++ compiler': 'inspect_file_as_text', 
    'access to google maps': 'ask_search_agent', 
    'youtube player': 'ask_search_agent', 
    'natural language processor': 'PythonInterpreter', 
    'graph interaction tools': 'inspect_file_as_text', 
    'bablyonian cuniform -> arabic legend': 'inspect_file_as_text', 
    'access to youtube': 'ask_search_agent', 
    'image search tools': 'ask_search_agent', 
    'calculator or counting function': 'PythonInterpreter', 
    'a speech-to-text audio processing tool': 'inspect_file_as_text', 
    'access to academic journal websites': 'ask_search_agent', 
    'pdf reader/extracter': 'inspect_file_as_text', 
    "rubik's cube model": 'inspect_file_as_text', 
    'wikipedia': 'ask_search_agent', 
    'video capability': 'visualizer', 
    'image processing tools': 'visualizer', 
    'image recognition software': 'visualizer', 
    'youtube': 'ask_search_agent'
}


def mapping(tools):
# write a function for me to replace the tools in the 'text' with ours
    mapped_tools = []
    for t in tools:
        mapped_tools.append(tool_map[t])
    return list(set(mapped_tools))

# Function to extract user content from JSON file
def extract_user_content_from_json(data):

    user_contents = []
    print ('datadatadatadata',data)
    count=0
    for case_data in data:
        tmp = {'query':"", "tools":[]}
        tools =  case_data["Annotator Metadata"]["Tools"]

        tmp_tool_list=[]
        tool_list=tools.split('\n')
        for tool in tool_list:
            index=tool.find('.')
            tool_temp=tool[index+1:].lower().lstrip(' ').rstrip(' ')
            tool_temp=remove_between_characters(tool_temp,'(',')').lstrip(' ').rstrip(' ')
            tmp_tool_list.append(tool_temp)

        tmp["query"] = case_data["Question"]
        tmp['tools'] = mapping(tmp_tool_list)
        print(f"---------------query-------------------{tmp['query']}")
        print ('before tools', tmp_tool_list)
        print(f"---------------tools-------------------{tmp['tools']}")
        user_contents.append(tmp) 
        count=count+1
        print ('countcountcountcount',count)

    return user_contents

# Function to sample in-context examples
def sample_in_context_examples(pool, num=10):
    # Randomly sample in-context examples from the pool
    print ('poolpoolpoolpool', pool)
    print ('numnumnumnum', num)
    return random.sample(pool, num)


# Function to format the prompt with in-context examples
# input:    
#      json_data: json data loaded from the path in json_prompt_paths
#      prompt: prompt loaded from the path in json_prompt_paths
def prompt_with_random_examples(json_data, prompt,num=10):
    user_contents = extract_user_content_from_json(json_data)
    in_context_examples = sample_in_context_examples(user_contents, num=num)
    in_context_examples = json.dumps(in_context_examples, indent=4)
    prompt = prompt.replace('IN_CONTEXT_EXAMPLES', "".join(in_context_examples))
    return prompt

# Function to load JSON and prompt from the paths provided
def load_json_prompt(json_prompt_paths):
    json_prompt = []
    for json_prompt_path in json_prompt_paths:
        with open(json_prompt_path['prompt'], 'r') as file:
            prompt = file.read()
        # with open(json_prompt_path['json'], 'r') as file:
        #     json_data = json.load(file)
        print ('json path', json_prompt_path['json'])
        json_data=read_jsonl(json_prompt_path['json'])
        json_prompt.append((json_data, prompt))

    return json_prompt


# call the gpt api to generate the query, using the prompt and json data
def fetch_system_prompts(mode=0, json_prompt_paths = json_prompt_paths, num=10, num_incontext_examples=10):
    # json_prompt = load_json_prompt(json_prompt_paths)
    # json_data, prompt = json_prompt[mode]
    system_prompts = []
    # print ('json_datajson_datajson_data',json_data)
    # print ('promptpromptpromptprompt',prompt)
    for i in range(num):
        json_prompt = load_json_prompt(json_prompt_paths)
        json_data, prompt = json_prompt[mode]
        system_prompt = prompt_with_random_examples(json_data, prompt,num_incontext_examples)
        system_prompts.append(system_prompt)
    return system_prompts
 
 # create the argument parser

def generate_identifier(length=16):
    characters = string.ascii_letters + string.digits + '_-!@#$%^&*'
    return ''.join(random.choices(characters, k=length))

def queries_to_json_and_save(queries, save_path):
    json_list = queries[7:-3]
    print(json_list)
    json_path = save_path + '/query/query_json/'
    json_list = json_list.replace('Query', 'query')
    json_list = json_list.replace('Tools', 'tools')
    json_list = json_list.replace('FileName', 'filename')
    # json_data = json.dumps(json_list, indent=4)
    folder = os.path.exists(save_path)
    if not folder:
        os.makedirs(json_path )
    
    json_string = f'[{json_list}]'

    # Step 2: Parse the JSON string
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    # Step 3: Write the JSON data to a file
    output_file = json_path + generate_identifier() + ".json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)   

    print(f"Data successfully saved to {output_file}")

    return output_file  


def save_to_json(path, json_file):
    with open(path, 'w') as f:
        json.dump(json_file, f)
def load_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file



def get_chat_response(messages, model=MODEL, temperature=0.2, max_tokens=2048, n=1, patience=10, sleep_time=2):
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

def get_queries(content: str, max_tokens, system_prompt):

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
        response = get_chat_response(messages=messages, model=MODEL, temperature=1, max_tokens=max_tokens)

    except openai.error.RateLimitError:
        print('openai.error.RateLimitError:')
        pass
    except Exception as e:
        print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
        pass
    return response
 
 
def multi_procfess_fetch_gpt_response(user_prompt, systen_prompt_incontext_example):

    print ('================systen_prompt_incontext_example',systen_prompt_incontext_example)
    print ('================user_prompt',user_prompt)

    try:
        response = get_queries(content=user_prompt, max_tokens=2048, system_prompt=systen_prompt_incontext_example)
        print ('responseresponseresponseresponse',response)
    except Exception as e:
        print(e)
        return None
    queries_to_json_and_save(response,args.save_path)
    return 1


def query_generation(args):

    user_prompt = "Please generate NUM_QUERIES queries. DO NOT output an id number before each query." 
    user_prompt = user_prompt.replace('NUM_QUERIES', str(args.np))
    # random sample ngpt of system prompts with different in-context examples
    system_prompts = fetch_system_prompts( num=args.ngpt, num_incontext_examples=args.ni, mode=args.mode)

    # multi_procfess_fetch_gpt_response(user_prompt, system_prompts[0])
 
    with ProcessPoolExecutor(max_workers = args.workers ) as executor:
        for system_prompt in tqdm(system_prompts):
             print(system_prompt)
             executor.submit(multi_procfess_fetch_gpt_response,user_prompt, system_prompt)

    # print(f"Query generation completed for {args.ngpt * args.np} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate queries using GAIA data')

    parser.add_argument("--number", type=int, default=3000)
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--save-path", type=str, default=f'data_generation/gaia_pipeline/save')
    parser.add_argument("--mode", type=int, default=0, help="0: single image no web, 1: single image web, 2: multi image no web, 3: multi image web")
    parser.add_argument('--output_path', type=str, help='Path to save the generated queries', default='query_generation/GAIA/gaia_queries.txt')
    parser.add_argument('--ngpt','--num_gpt_queries', type=int, help='Number of queries to generate', default=100)
    parser.add_argument('--ni','--num_incontext_examples', type=int, help='Number of in-context examples to include in the prompt', default=20)
    parser.add_argument('--np','--query_num_per_gpt_call', type=int, help='Number of queries to generate per GPT call', default=10)
    args = parser.parse_args()

    timestamp=args.timestamp
    args.save_path=f'data_generation/gaia_pipeline/save/{timestamp}'


    # redirect the output to a log file 
    os.makedirs(f'data_generation/gaia_pipeline/log/{timestamp}/', exist_ok=True)
    sys.stdout = open(f'data_generation/gaia_pipeline/log/{timestamp}/{timestamp}_0_query.log', 'a')

    print(f"GAIA based Query GENERATION STARTED:")
    query_generation(args)
    merge(args.save_path, timestamp)


