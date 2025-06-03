import json
from tqdm import tqdm
# GTA
data_path = "/scratch/zhangbofei/Projects/Multimodal-CL/iclr_09/TongAgent/experiments/CPM-FT/data/agent_tune_dataset_cpm_8k_gta_with_verifier.json"
# GAIA
data_path = "/scratch/zhangbofei/Projects/Multimodal-CL/iclr_09/TongAgent/experiments/CPM-FT/data/agent_tune_dataset_cpm_17k_gaia_with_verifier.json"


with open(data_path, "r") as f:
    dataset = json.load(f)

def _convert(image_path_map, conversations):
    output = []
    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        turn_new = dict()
        turn_new["from"] = role
        pid = 1
        keys = sorted(list(image_path_map.keys()))
        for k in keys:
            v = image_path_map[k]
            if k in content:
                content = content.replace(k, f"Picture {pid}: <img>{v}</img>\n")
                content = content.replace(f"</img>\n\n", "</img>\n")
                pid += 1
        turn_new["value"] = content
        output.append(turn_new)
    return output


for item in tqdm(dataset):
    #print(item["image"])
    #print(item.keys())
    conversations = item["conversations"]
    #print(len(conversations), conversations[1])
    image_path_map = dict()
    if type(item["image"]) == str:
        image_path_map["<image>"] = item["image"]
    else:
        for k, v in item["image"].items():
            image_path_map[k] = v
    item["conversations"] = _convert(image_path_map, conversations)

from datetime import datetime
import json

now = datetime.now().strftime("%Y%m%d_%H%M")
print("write to", f"data/train_{now}.json")
with open(f"data/train_{now}.json", "w") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

import random
with open(f"data/train_{now}_subset.json", "w") as f:
    random.shuffle(dataset)
    json.dump(dataset[:1000], f, indent=4, ensure_ascii=False)