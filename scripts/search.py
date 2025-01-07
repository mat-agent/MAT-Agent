import json
data_path = "experiments/CPM-FT/data/agent_tune_dataset_gaia_1206_11k.json"
with open(data_path, "r") as f:
    data = json.load(f)
    
search_str = '''The attached file contains a list of vendors in the Liminal Springs mall, along with each vendor’s monthly revenue and the rent they pay the mall. I want you to find the vendor that makes the most money, relative to the rent it pays. Then, tell me what is listed in the “type” column for that vendor.'''
found = False
for item in data:
    # print(item)
    conversations = item["conversations"]
    for conversation in conversations:
        if search_str in conversation["content"]:
            print(item)
            found = True
            
            
print(found)

