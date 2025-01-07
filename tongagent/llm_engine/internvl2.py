import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

def load_pretrain_model():
    
    path = 'OpenGVLab/InternVL2-8B'
    print("create model from", path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def examples():
    model, tokenizer = load_pretrain_model()

    # set the max number of tiles in `max_num`
    pixel_values = load_image('tests/data/airplane.jpeg', max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # pure-text conversation (纯文本对话)
    question = 'Hello, who are you?'
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    # single-image multi-round conversation (单图多轮对话)
    question = '<image>\nPlease describe the image in detail.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    print("History:", history)
    


class ModelSingleton():
    def __new__(cls, model_name, lora_path=None):
        if hasattr(cls, "model"):
            return cls

        model, tokenizer = load_pretrain_model()
        cls.model = model
        cls.tokenizer = tokenizer
        return cls
    
    
from transformers.agents.llm_engine import MessageRole, HfApiEngine, get_clean_message_list

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
    # MessageRole.SYSTEM: MessageRole.USER
}

from typing import Optional
class InternVL2Engine(HfApiEngine):
    def __init__(self, model_name: str = "", lora_path: Optional[str] = None):
        module = ModelSingleton(model_name, lora_path)
        self.has_vision = True
        model, tokenizer = module.model, module.tokenizer
        self.model, self.tokenizer = model, tokenizer
        self.model_name = model_name
    
    def __call__(self, messages, stop_sequences=[], *args, **kwargs) -> str:
        # print ('----------------raw message',messages)
        torch.cuda.empty_cache()
        image_paths = kwargs.get("image_paths", None)
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
        if len(msgs) >= 2:
            first_msg = msgs[0]
            second_msg = msgs[1]
            if first_msg["role"] == "system" and second_msg["role"] == "user":
                # do merge
                msg = {
                    "role": "user",
                    "content": first_msg["content"] + "--------\n" + second_msg["content"]
                }
                if image_paths is not None and len(image_paths) > 0 and not msg["content"].startswith("<image>"):
                    msg["content"] = "<image>\n" + msg["content"]

                msgs = [msg] + msgs[2:]
        
        # do completion
        pixel_values = []
        if image_paths is not None and len(image_paths) > 0:
            for image_path in image_paths:
                pixel_values.append(load_image(image_path))
            pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            pixel_values = None
            
        
        history = []
        for msg_idx in range(0, len(msgs), 2):
            if msg_idx + 1 >= len(msgs):
                break
            history.append((msgs[msg_idx]["content"], msgs[msg_idx + 1]["content"]))
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response, history = self.model.chat(self.tokenizer, pixel_values, msgs[-1]["content"], generation_config, history=history, return_history=True)
        answer = response
        # print(answer)
        for stop in stop_sequences:
            stop_idx = answer.find(stop)
            if stop_idx == -1:
                continue
            answer = answer[:stop_idx]
        return answer
       
            
        