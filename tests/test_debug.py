import json

with open("debug.json", "r") as f:
    data = json.load(f)
    

for item in data:
    conversations = item["conversations"]
    for conversation in conversations:
        if conversation["role"] == "user":
            if '.png' in conversation["content"]:
                print(conversation["content"])
                print("-" * 10)
            