from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list
from tongagent.utils import load_config
import re

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
from openai import OpenAI

import torch

def load_pretrained_model(model_name):
    torch.manual_seed(0)
    print("from pretrained", model_name)
    if "VL" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_client_model(endpoint):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{endpoint}:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

class ModelSingleton():
    def __new__(cls, model_name, lora_path=None):
        if hasattr(cls, "model_dict") and model_name in cls.model_dict:
            return cls

        if not hasattr(cls, "model_dict"):
            cls.model_dict = dict()
            
        if "VL" in model_name:
            model, tokenizer = load_pretrained_model(model_name)
            if lora_path is not None:
                print("Load Qwen-VL from lora", lora_path)
                import time
                from peft.peft_model import PeftModel
                time.sleep(10)
                model = PeftModel.from_pretrained(model, lora_path)
                model.merge_and_unload()
            cls.model_dict[model_name] = (model, tokenizer)
            
        else:
            config = load_config()
            model = load_client_model(config.qwen.endpoint)
            tokenizer = None
            cls.model_dict[model_name] = (model, tokenizer)
        return cls

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    # MessageRole.SYSTEM: MessageRole.USER
}

from typing import Optional
class QwenEngine(HfApiEngine):
    def __init__(self, model_name: str = "", lora_path: Optional[str] = None):
        module = ModelSingleton(model_name, lora_path)
        self.has_vision = False
        model, tokenizer = module.model_dict[model_name]
        if 'VL' in model_name:
            self.has_vision = True
            self.processor = tokenizer # for VLM use processor as tokenizer
            
        self.model, self.tokenizer = model, tokenizer
        self.model_name = model_name
    def call_llm(self, messages, stop_sequences=[], *args, **kwargs):
        assert not self.has_vision, "Should use this function with Qwen LLM"
        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=512,
        #     temperature=0.7,
        #     top_p=0.8,
        #     top_k=100,
        #     do_sample=True,
        #     repetition_penalty=1.05
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return answer
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
        )        
        return response.choices[0].message.content

    def call_vlm(self, messages, stop_sequences=[], *args, **kwargs):
        print("call vlm")
        assert self.has_vision, "Should use this function with Qwen VL model"
        image_paths = kwargs.get("image_paths", [])
        for msg_id, msg in enumerate(messages):
            if msg["role"] == "user":
                content_replace = []
                if len(image_paths) > 0:
                    for image_path in image_paths:
                        content_replace.append({
                            "type": "image",
                            "image": image_path
                        })
                    
                content = {"type": "text", "text": msg["content"]}
                content_replace.append(content)
                messages[msg_id] = {
                    "role": "user",
                    "content": content_replace
                }
                break
        print("msg=", messages)
        text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=100,
            do_sample=True,
            repetition_penalty=1.05
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        torch.cuda.empty_cache()
        image_paths = kwargs.get("image_paths", [])
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
        if not self.has_vision:
            answer = self.call_llm(messages, stop_sequences=stop_sequences)
        else:
            answer = self.call_vlm(messages, stop_sequences=stop_sequences, image_paths=image_paths)
        # print(answer)
        for stop in stop_sequences:
            stop_idx = answer.find(stop)
            if stop_idx == -1:
                continue
            answer = answer[:stop_idx]
        return answer
        
if __name__ == "__main__":
    model, tokenizer = load_pretrained_model("Qwen/Qwen2.5-7B-Instruct")
    print(model)