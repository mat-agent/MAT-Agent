import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True)

args = parser.parse_args()
subfolders = os.listdir(args.folder)

total = 0
correct = 0
all_samples = len(subfolders)
for subfolder in subfolders:
    data_path = os.path.join(args.folder, subfolder, "agent_memory.json")
    
    with open(data_path, "r") as f:
        dataset = json.load(f)
    
    gt, answer = dataset["ground_truth"], dataset["final_answer"]
    if gt is None:
        continue
    skip = False
    is_correct = True
    for each in gt:
        if type(each) is str:
            skip = True
            break
        
        if type(each) is list:
            is_this_gt_correct = []
            for item in each:
                is_this_gt_correct.append(item.lower() in str(answer).lower())
            
            is_correct = is_correct and any(is_this_gt_correct)
        else:
            raise ValueError("unexpected")
            
    if skip:
        continue
    if is_correct:
        print("Correct:", gt, answer)
        correct += 1
    else:
        print("Incorrect", gt, answer)
    total += 1
    # print(gt, answer)

print("Folder", args)
print("Total samples valid:", total, "Correct sample", correct, "all samples", all_samples)
print("Accuracy", round(correct / total, 4) * 100, "%")
print("Accuracy (all samples)", round(correct / all_samples, 4) * 100, "%")