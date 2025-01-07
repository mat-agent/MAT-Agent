# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

def load(model_name):
    if 'llava-onevision-qwen2-7b-ov-chat' in model_name:
        pretrained = 'lmms-lab/llava-onevision-qwen2-7b-ov-chat'
        model_name = 'llava_qwen'
        conv_template = 'qwen_1_5'
    elif 'llama3-llava-next-8b' in model_name:
        pretrained = 'lmms-lab/llama3-llava-next-8b'
        model_name = 'llava_llama_3'
        conv_template = 'llava_llama_3'
    elif 'open-llava-next-llama3-8b' in model_name:
        pretrained = 'Lin-Chen/open-llava-next-llama3-8b'
        model_name = 'llava_llama_3'
        conv_template = 'llava_llama_3'
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    model.eval()
    return tokenizer, model, image_processor, max_length, conv_template

class ModelSingleton:
    def __new__(cls, model_name):
        if hasattr(cls, "model_name"):
            return cls
        cls.model_name = model_name
        cls.tokenizer, cls.model, cls.image_processor, cls.max_length, cls.conv_template = load(model_name)
        return cls
    
openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    # MessageRole.SYSTEM: MessageRole.USER
}

class LLaVAEngine(HfApiEngine):
    def __init__(self, model_name):
        module = ModelSingleton(model_name)
        self.model_name = model_name
        self.tokenizer = module.tokenizer
        self.model = module.model
        self.image_processor = module.image_processor
        self.max_length = module.max_length
        self.conv_template = module.conv_template
        
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        print(f"Using model as engine: {self.model_name}")
        # print ('----------------raw message',messages)
        torch.cuda.empty_cache()
        image_paths = kwargs.get("image_paths", [])
        if len(image_paths) > 1:
            image_paths = image_paths[:1]
            
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        #print ('----------------processed message',messages)
        task = messages[0]
        msgs = []
        for msg in messages:
            # print(msg["role"].value)
            if msg["role"] == MessageRole.SYSTEM:
                msgs.append(
                    {
                        "role": "system",
                        "content": msg["content"]
                    }
                )
            else:
                msgs.append(
                    {
                        "role": "user" if msg["role"] == MessageRole.USER else "assistant",
                        "content": msg["content"]
                    }
                )
        # do merge if first and second message are systems and users
        if len(msgs) > 1 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user":
            first_msg = msgs[0]["content"] + msgs[1]["content"]
            first_msg = {"role": "user", "content": first_msg}
            msgs = [first_msg] + msgs[2:]
        conv = copy.deepcopy(conv_templates[self.conv_template])
        for msg_id, msg in enumerate(msgs):
            role_idx_mapping = {
                "user": 0,
                "assistant": 1,
            }
            role_idx = role_idx_mapping[msg["role"]]
            content = msg["content"]
            if msg_id == 0 and DEFAULT_IMAGE_TOKEN not in content and len(image_paths) > 0:
                content = DEFAULT_IMAGE_TOKEN + "\n" + content
            conv.append_message(conv.roles[role_idx], content)
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        if len(image_paths) > 0:
            image = Image.open(image_paths[0])
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensor]
            image_sizes = [image.size]
        else:
            image_tensor = None
            image_sizes = None
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        answer = text_outputs[0]
        # print(answer)
        for stop in stop_sequences:
            stop_idx = answer.find(stop)
            if stop_idx == -1:
                continue
            answer = answer[:stop_idx]
        return answer
