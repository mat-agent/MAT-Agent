from tongagent.llm_engine.mini_cpm import load_pretrained_model_lora
import json
from PIL import Image

model, tokenizer = load_pretrained_model_lora("experiments/CPM-FT/output/cpm_v2_6_7882650_2024_10_14_19_25/")
input_data = ".cache/gta/cpm_v2_6_7882650_2024_10_14_19_25/0/agent_memory.json"
image_paths = ["data/gta_dataset/image/image_1.jpg", "data/gta_dataset/image/image_2.jpg"]


with open(input_data, "r") as f:
    data = json.load(f)
messages = data["conversations"]
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

system_prompt = messages[0]["content"]
print("prompt", messages[1:2])
answer = model.chat(
    image=None,
    msgs=messages[1:2],
    system_prompt=system_prompt,
    tokenizer=tokenizer
)
print(answer)