from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
ds = load_dataset("BAAI/Infinity-Instruct", "0625")

ds = ds["train"]
selected = [
    "python programming",
    #"search skills",
    #"code refactoring",
    #"search engine optimization",
    #"code debugging",
    #"code modification",
    #"code implementation",
]
import copy
def process(item):
    item_new = copy.deepcopy(item)
    item_new["image"] = dict()
    
    conv = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        conv.append(
            {"role": role, "content": turn["value"]}
        )
    item_new["conversations"] = conv
    return item_new
saved = []
for item in tqdm(ds):
    abs = item["label"]["ability_en"]
    keep = False
    for each in abs:
        if each in selected:
            keep = True
            break
        
        
    
    if not keep:
        continue
    
    saved.append(process(item))

print("Total", len(saved))   
with open("subset.json", "w") as f:
    import json
    json.dump(saved, f, indent=4, ensure_ascii=False)