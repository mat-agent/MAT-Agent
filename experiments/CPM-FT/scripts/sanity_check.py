from finetune.dataset import conversation_to_ids_qwen2
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


import json
import torch
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data-file", required=True)
args = parser.parse_args()

source_file = args.data_file
with open(source_file, "r") as f:
    dataset = json.load(f)

max_length = 10240 
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
filter_dataset = []
token_lens = []
for item in tqdm(dataset):
    # item = 
    # conv = [{'role': 'user', 'content': '<image></image>Task: Describe this image.\nAttachment: data/tongagent/coco/train2017/000000096046.jpg;'}, {'role': 'assistant', 'content': 'Thought: I will use the `image_generator` tool to create a new image of a doughnut with rainbow sprinkles set in a coffee shop environment. By crafting a detailed prompt, I can ensure the generated image meets the requirements.\n\nCode:\n```py\nprompt = "A fresh doughnut with rainbow sprinkles on a white plate, placed next to a steaming cup of coffee in a cozy coffee shop setting, with wooden tables and warm lighting."\nimage_path = image_generator(prompt=prompt)\nprint(image_path)\n```'}, {'role': 'user', 'content': '[OUTPUT OF STEP 0] Observation:\n.cache/1727238726097_rfp28nah.jpeg\n'}, {'role': 'assistant', 'content': 'Thought: The image of the doughnut with rainbow sprinkles in a coffee shop setting has been successfully generated and saved. Now, I will provide the path to this generated image as the final output.\n\nCode:\n```py\nfinal_answer("Generated image saved at: .cache/1727238726097_rfp28nah.jpeg")\n```'}]
    conv = item["conversations"]
    system_prompt = None if conv[0]["role"] != "system" else conv[0]["content"]
    if system_prompt:
        conv = conv[1:]
    try:
        input_ids, context, raw_msg = conversation_to_ids_qwen2(
            conv,
            tokenizer,
            system_prompt=system_prompt
        )
    except Exception as e:
        print(repr(e), "have to skip")
        print(item["id"])
        # print(item["conversations"])
        continue
    
    if "correct" in item:
        token_lens.append(input_ids.shape[-1])
    if input_ids.shape[-1] > max_length:
        print("overflow!")
        print(input_ids.shape[-1])
        print(item["id"])
        continue
    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))
    if torch.all(context):
        raise Exception("No tokens available to compute loss.")
    
    
    filter_dataset.append(item)
print("Size", len(dataset), len(filter_dataset))

with open(source_file.replace('agent_tune_dataset_cpm', "agent_tune_dataset_cpm_cleaned"), "w") as f:
    json.dump(filter_dataset, f, indent=4, ensure_ascii=False)

with open("token_len.json", "w") as f:
    json.dump(token_lens, f, indent=4, ensure_ascii=False)