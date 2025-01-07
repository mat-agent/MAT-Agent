
import json

with open("data/gta_6350_merged.json", "r") as f:
    dataset = json.load(f)
    
for item in dataset:
    if item["id"] == "vB9O_XTo":
        print(item["image"])
        conv = item["conversations"]
        for turn in conv:
            print(turn["role"])
            print(turn["content"])
            print("-" * 100)
        break


