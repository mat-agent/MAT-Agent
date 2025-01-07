from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list
from tongagent.utils import load_config
import re
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from peft.peft_model import PeftModel

def load_pretrained_model(model_id = None):
    if model_id is None:
        model_id = "openbmb/MiniCPM-V-2_6"
    torch.manual_seed(0)

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def load_pretrained_model_lora(peft_model_id):
    model, tokenizer = load_pretrained_model()
    print("Load Lora")
    model = PeftModel.from_pretrained(model, peft_model_id)
    print("Lora merge and unload")
    model.merge_and_unload()
    return model, tokenizer

class ModelSingleton():
    def __new__(cls, model):
        
            
        if hasattr(cls, "model"):
            return cls.model, cls.tokenizer

        if model is not None:
            if os.path.exists(os.path.join(model, "adapter_config.json")):
                print("Load Lora!")
                cls.model, cls.tokenizer = load_pretrained_model_lora(model)
            else:
                print("Load full parameters!")
                cls.model, cls.tokenizer = load_pretrained_model(None)
        else:
            cls.model, cls.tokenizer = load_pretrained_model(None)
        return cls

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    # MessageRole.SYSTEM: MessageRole.USER
}

class MiniCPMEngine(HfApiEngine):
    def __init__(self, model=None, disable_vision=False):
        module = ModelSingleton(model)
        self.model, self.tokenizer = module.model, module.tokenizer
        self.disable_vision = disable_vision
        
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        image_paths = kwargs.get("image_paths", [])
        if self.disable_vision:
            if len(image_paths) > 0:
                print("Warning: Vision is disabled, but image paths are provided.")
            else:
                image_paths = []
            
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        #print ('----------------processed message',messages)

        task = messages[0]
        content = task['content']
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
            messages[1]['content'] = []
            messages[1]['content'].append(dict(type="text", text=origin_content))
            prompt = []
            for path_item in path_list_new:
                # messages[1]['content'].append(dict(type="image_url", image_url={"url": f"data:image/jpeg;base64,{encode_image(path_item)}"}))
                image = Image.open(path_item).convert('RGB')
                prompt.append(image)
            prompt.append(origin_content)
            messages[1]["content"] = prompt
                
        if image_paths is not None and len(image_paths) > 0:
            origin_content = messages[1]['content']
            messages[1]['content'] = []
            messages[1]['content'].append(dict(type="text", text=origin_content))
            prompt = []
            for path_item in image_paths:
                image = Image.open(path_item).convert('RGB')
                prompt.append(image)
            prompt.append(origin_content)
            messages[1]["content"] = prompt
                
        # print(messages)
        
        system_prompt = None
        msgs = []
        for msg in messages:
            # print(msg["role"].value)
            if msg["role"] == MessageRole.SYSTEM:
                system_prompt = msg["content"]
            else:
                msgs.append(
                    {
                        "role": "user" if msg["role"] == MessageRole.USER else "assistant",
                        "content": msg["content"]
                    }
                )
        # exit()
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            system_prompt=system_prompt,
            tokenizer=self.tokenizer
        )
        # print(answer)
        for stop in stop_sequences:
            stop_idx = answer.find(stop)
            if stop_idx == -1:
                continue
            answer = answer[:stop_idx]
        return answer