import sys
sys.path.insert(0, "./")
import sqlite3
print("Importing tongagent")
from tongagent.agents.data_sampling_agent import create_agent
from tongagent.utils import load_config
from datasets import load_dataset
import os
import argparse
from typing import Optional
from tqdm import tqdm
import ray
print("Importing ray")
ray.init()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--engine',
    '-e',
    choices=["minicpm", 'tonggpt', 'qwen', "internvl2", "llava"],
    default="tonggpt"
)

parser.add_argument(
    '--lora-path',
    '-lp',
    default=None
)

parser.add_argument(
    '--data-name',
    choices=["2023_level1", "2023_level2", "2023_level3"]
)

parser.add_argument(
    '--split',
    choices=["validation", "test"]
)
args = parser.parse_args()

DATA_NAME = args.data_name
SPLIT = args.split

try:
    import wandb
    wandb.init(project="MAT_Evaluation", name=f"eval_gaia")
    wandb.alert(title="Evaluation started", text=f"Evaluation started for {DATA_NAME} {SPLIT}")
except:
    print("Exception!")
    pass

# @ray.remote(num_gpus=1)
class GAIAAgent():
    def __init__(self) -> None:
        os.makedirs(f".cache/qa_cache/{SPLIT}", exist_ok=True)
        os.makedirs(f".cache/qa_cache/{SPLIT}/{args.engine}", exist_ok=True)
        self.config = load_config()
        if args.engine in ["tonggpt"]:
            self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, self.config.tonggpt.model_name)
        elif args.engine in ["qwen"]:
            if args.lora_path is not None:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, args.lora_path)
            else:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, self.config.qwen.model_name)
        elif args.engine in ["internvl2"]:
            if args.lora_path is not None:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, args.lora_path)
            else:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, self.config.internvl2.model_name)
        elif args.engine in ["minicpm"]:
            if args.lora_path is not None:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, args.lora_path)
            else:
                self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, self.config.minicpm.model_name)
        elif args.engine in ["llava"]:
            self.qa_cache_db = os.path.join(".cache/qa_cache/", SPLIT, args.engine, self.config.llava.model_name)
            
        os.makedirs(self.qa_cache_db, exist_ok=True)
        qa_cache_db_path = os.path.join(self.qa_cache_db, f"{DATA_NAME}.db")
        self.cache_root = self.qa_cache_db
        self.qa_cache_db_path = qa_cache_db_path
        qa_cache_db = sqlite3.connect(qa_cache_db_path)
        qa_cache_db.execute('''CREATE TABLE IF NOT EXISTS qa_cache
                (question TEXT PRIMARY KEY     NOT NULL,
                task_id TEXT NOT NULL,
                answer           TEXT    NOT NULL,
                groundtruth TEXT);''')
        qa_cache_db.commit()
        qa_cache_db.close()
        
        # self.agent = create_agent_simple_gaia()
        self.agent = create_agent(args.engine, "gaia", 5, args.lora_path)
    def ask(self, raw_question: str, attachment_name: Optional[str], task_id: str, groundtruth: Optional[str]):
        cache_db = sqlite3.connect(self.qa_cache_db_path)
        cursor = cache_db.cursor()
        cursor.execute(f"SELECT answer FROM qa_cache WHERE question = ?", (raw_question,))
        row = cursor.fetchone()
        cache_db.close()

        if row is not None:
            print(f"Cache hit for question: {raw_question}")
            print("Cache hit:", row)
            return row[0]
        else:
            print(f"Cache miss for question: {raw_question}")
        if attachment_name is not None and attachment_name.strip() != "":
            question = f"{raw_question}\nAttachment: data/GAIA/2023/{SPLIT}/{attachment_name}"
        else:
            question = raw_question
        
        if attachment_name is not None and (attachment_name.endswith(".png") or attachment_name.endswith(".jpg")):
            self.agent.image_paths = [f'data/GAIA/2023/{SPLIT}/{attachment_name}']
        else:
            self.agent.image_paths = []
            
        result = self.agent.run(question)
        try:
            saved_path = os.path.join(self.cache_root, task_id)
            path = self.agent.save_trajectory(path=saved_path, ground_truth=groundtruth, final_answer=result)    
            print("save", task_id, result, path)
        except Exception as e:
            print("save failed!", repr(e))
            
            
        try:
            cache_db = sqlite3.connect(self.qa_cache_db_path)
            cursor = cache_db.cursor()
            cursor.execute("INSERT INTO qa_cache (question,answer, task_id, groundtruth) VALUES (?, ?, ?, ?)", (raw_question, result, task_id, groundtruth))
            cache_db.commit()
            cache_db.close()
        except Exception as e:
            print(f"Ignoring error: {e} when inserting question: {question}")
        return result
    
@ray.remote(num_gpus=1)
def worker(items):
    agent = GAIAAgent()
    n_data = len(items["Question"])
    for i in tqdm(range(n_data)):
        print("Item", i)
        question = items["Question"][i]
        file_name = items["file_name"][i]
        task_id = items["task_id"][i]
        groundtruth = items["Final answer"][i] if 'Final answer' in items else ""
        result = agent.ask(
            question,
            file_name,
            task_id,
            groundtruth
        )
        print("#" * 100)
        print(question)
        print(result)
        print("#" * 100)
            

def main():
    ds = load_dataset("gaia-benchmark/GAIA", DATA_NAME, split=SPLIT)
    agent = GAIAAgent()
    # agent_pool = ray.util.ActorPool([GAIAAgent.remote() for _ in range(num_worker)])
    # answers = list(agent_pool.map(lambda agent, row: agent.ask.remote(row['Question'], row['file_name'], row["task_id"], row["Final answer"]), ds))
    # print('answers', answers)
    for item in tqdm(ds):
        print("Item", item)
        question = item["Question"]
        file_name = item["file_name"]
        task_id = item["task_id"]
        groundtruth = item["Final answer"]
        result = agent.ask(
            question,
            file_name,
            task_id,
            groundtruth
        )
        print("#" * 100)
        print(question)
        print(result)
        print("#" * 100)

import torch
if __name__ == "__main__":
    num_worker = torch.cuda.device_count()
    if num_worker == 1:
        main()
    else:
        ds = load_dataset("gaia-benchmark/GAIA", DATA_NAME, split=SPLIT)
        futures = []
        batch_size = len(ds) // num_worker
        for i in range(num_worker):
            start = i * batch_size
            end = (i + 1) * batch_size
            if i == num_worker - 1:
                end = len(ds)
            
            items = ds[start:end]
            print("items", items)
            futures.append(worker.remote(items))
        results = ray.get(futures)
        
        