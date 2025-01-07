import sys
import json
sys.path.insert(0, "./")

from tongagent.agents.data_sampling_agent import create_agent
from tongagent.utils import load_config
from tqdm import tqdm
import os
import argparse 
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    '-e',
    choices=["minicpm", 'tonggpt', "qwen", "internvl2", "llava"],
    default="tonggpt"
)

parser.add_argument(
    '--lora-path',
    '-lp',
    default=None
)

parser.add_argument(
    "--disable-vision",
    action="store_true"
)
args = parser.parse_args()
print("Start run", args.engine)
with open("data/gta_dataset/dataset.json", "r") as f:
    dataset = json.load(f)
    
# print(dataset[0])
llm_engine = args.engine
config = load_config()
if args.lora_path is None:
    if args.engine == "tonggpt":
        root = f".cache/gta/{config.tonggpt.model_name}"
    elif args.engine == "internvl2":
        root = f".cache/gta/{config.internvl2.model_name}"
    elif args.engine == "llava":
        root = f".cache/gta/{config.llava.model_name}"
    else:
        root = f".cache/gta/{config.qwen.model_name}"
else:
    if "checkpoint-" in args.lora_path:
        path = args.lora_path.strip('/').split("/")
        path = path[len(path)-2:]
        path = "_".join(path)
        root = f".cache/gta/{path}"
    else:
        root = os.path.join(".cache/gta/", args.lora_path.strip("/").split("/")[-1])

if args.disable_vision:
    root += "_without_vision"
print("save to", root)
os.makedirs(root, exist_ok=True)
try:
    import wandb
    wandb.init(project="MAT_Evaluation", name=f"eval_gta")
    wandb.alert(title="Evaluation started", text=f"Evaluation started for {args.engine}")
except:
    print("Exception!")
    pass
import ray
ray.init()


def worker(keys, dataset):
    agent = create_agent(
        llm_engine=llm_engine,
        task="gta",
        error_tolerance=5,
        lora_path=args.lora_path,
        disable_vision=args.disable_vision
    )
    for k in tqdm(keys):
        item = dataset[k]
        question = item["dialogs"][0]["content"].strip()
        image_paths = [os.path.join("data/gta_dataset",i["path"]) for i in item["files"]]
        gt = item["gt_answer"]["whitelist"] if type(item["gt_answer"]) is dict else item["gt_answer"]
        saved_path = os.path.join(root, k)
        if os.path.exists(saved_path):
            print("already done so skip")
            continue
        
        print(k, question, image_paths, gt)
        if len(image_paths) > 0:
            suffix = "\nAttachment:\n"
            suffix += '\n'.join(image_paths)
            question += suffix
        
        if args.disable_vision:
            agent.set_image_paths([])
        else:
            agent.set_image_paths(image_paths)
        result = agent.run(question)
        
        path = agent.save_trajectory(path=saved_path, ground_truth=gt, final_answer=result)    
        print("save", k, result, path)

@ray.remote(num_gpus=1)
def remote_worker(keys, dataset):
    worker(keys, dataset)
    
    
n_total_gpu = torch.cuda.device_count()
n_total_data = len(dataset)
data_keys = list(dataset.keys())
futures = []

if n_total_gpu == 1:
    worker(data_keys, dataset)
else:
    batch_size = n_total_data // n_total_gpu
    random.shuffle(data_keys)
    for i in range(n_total_gpu):
        start = i * batch_size
        end = (i + 1) * batch_size
        if i == n_total_gpu - 1:
            end = len(data_keys)
        f = remote_worker.remote(data_keys[start:end], dataset)
        futures.append(f)
    
print(ray.get(futures))    
ray.shutdown()