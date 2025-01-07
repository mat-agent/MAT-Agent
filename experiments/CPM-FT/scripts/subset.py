
import json


source_file = "data/agent_tune_dataset_cpm.json"
with open(source_file, "r") as f:
    dataset = json.load(f)
    
    
with open("data/debug_small.json", "w") as f:
    json.dump(dataset[:100], f, indent=4, ensure_ascii=False)