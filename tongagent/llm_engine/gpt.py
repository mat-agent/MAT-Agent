from openai import AzureOpenAI, RateLimitError, OpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list, HfApiEngine
from tongagent.utils import load_config
import base64
import re

def extract_and_delete(str1, str2, str3):
    # Find the starting index of str1 in str3
    start_index = str3.find(str1)
    if start_index == -1:
        return None, str3  # str1 not found in str3
    start_index += len(str1)  # Move to the end of str1
    # Find the ending index of str2 in str3 starting from the end of str1
    end_index = str3.find(str2, start_index)
    if end_index == -1:
        return None, str3  # str2 not found in str3 after str1
    # Extract the substring between str1 and str2
    extracted = str3[start_index:end_index]
    # Remove the extracted substring (including str1 and str2) from str3
    new_str3 = str3[:start_index - len(str1)] + str3[end_index + len(str2):]
    return extracted, new_str3

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_tonggpt_open_ai_client(model="tonggpt"):
    config = load_config()
    gpt_config = getattr(config, model)
    print("Init gpt client by config:", gpt_config)
    if gpt_config.open_ai_client_type == "azure":
        endpoint = f"https://api.tonggpt.mybigai.ac.cn/proxy/{gpt_config.region}"
        return AzureOpenAI(
                api_key=gpt_config.api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
        ), gpt_config.model_name
    else:
        return OpenAI(api_key=gpt_config.api_key), gpt_config.model_name

    
openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class TongGPTEngine(HfApiEngine):
    def __init__(self, model="tonggpt"):
        client, model_name = get_tonggpt_open_ai_client(model)
        self.model_name= model_name
        self.client = client
        
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        image_paths = kwargs.get("image_paths", None)
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        print ('----------------processed message',messages)
        if len(messages) > 1:
            task = messages[1]
            content = task['content']
        elif len(messages) >= 1:
            for msg in messages:
                if msg['role'] == "user":
                    task = msg["content"]
                    content = msg['content']
                    break
        else:
            raise Exception("No messages found")
        # print ('contentcontentcontent',content)

        if 'Attached image ' in content:
            match = re.search(r'Attached image (\d+) paths: ', content)
            if match:
                number = int(match.group(1))
                origin_content = content[:match.start()]
                paths = content[match.end():].split('; ')
                path_list_new = paths[:number]
            else:
                origin_content = content
                path_list_new = []

            # print ('path_listpath_listpath_list',path_list_new)
            messages[1]['content']=[]
            messages[1]['content'].append(dict(type="text", text=origin_content))

            for path_item in path_list_new:
                if 'png' in path_item or 'jpg' in path_item or 'jpeg' in path_item:
                    messages[1]['content'].append(dict(type="image_url", image_url={"url": f"data:image/jpeg;base64,{encode_image(path_item)}"}))
                
        if image_paths is not None and len(image_paths) > 0:
            origin_content = messages[1]['content']
            messages[1]['content'] = []
            messages[1]['content'].append(dict(type="text", text=origin_content))
            
            for path_item in image_paths:
                messages[1]['content'].append(dict(type="image_url", image_url={"url": f"data:image/jpeg;base64,{encode_image(path_item)}"}))
                
        #print ('messagesmessagesmessages',messages[0])
        #print ('messagesmessagesmessages1111',messages[1])
        retry = 3
        for i in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stop=stop_sequences,
                )
                break
            except RateLimitError as e:
                print("catch rate limit error")
                import time
                time.sleep(10)
            
        return response.choices[0].message.content

