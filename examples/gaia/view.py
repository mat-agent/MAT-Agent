import sys
sys.path.insert(0, "./")
import sqlite3

from tongagent.evaluation.gaia_scorer import question_scorer
from tongagent.llm_engine.gpt import get_tonggpt_open_ai_client
from tongagent.prompt import FORMAT_ANSWER_PROMPT_GAIA
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
args = parser.parse_args()

# cache_db = sqlite3.connect(args.data_path)
# cursor = cache_db.cursor()
# cursor.execute(f"SELECT * FROM qa_cache")
# rows = cursor.fetchall()
# print(rows)
# print(len(rows))

from datasets import load_dataset
ds = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")

subset = ds[0:10]
for k, v in subset.items():
    print(k, len(v))
