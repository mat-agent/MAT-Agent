from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
ds = load_dataset("BAAI/Infinity-Instruct", "0625")

ds = ds["train"]
ability = defaultdict(lambda : 0)
cate_ability = defaultdict(lambda : 0)
for item in tqdm(ds):
    # print(item)
    abs = item["label"]["ability_en"]
    for each in abs:
        ability[each] += 1
    
    abs = item["label"]["cate_ability_en"]
    for each in abs:
        cate_ability[each] += 1
    # break
print(ability)
print(cate_ability)

with open("stats.json", "w") as f:
    import json
    json.dump({"abs": ability, "cate_abs": cate_ability}, f, indent=4, ensure_ascii=False)