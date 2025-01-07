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

cache_db = sqlite3.connect(args.data_path)
cursor = cache_db.cursor()
cursor.execute(f"SELECT * FROM qa_cache")
rows = cursor.fetchall()
cache_db.close()

# print(rows)
client, model = get_tonggpt_open_ai_client()
template = ChatPromptTemplate.from_template(FORMAT_ANSWER_PROMPT_GAIA)
n_total = len(rows)
correct = 0
eval_data = []
for row in tqdm(rows):
    try:
        is_this_correct = question_scorer(
            ground_truth=row[-1],
            model_answer=row[-2]
        )
    except Exception as e:
        print("question_scorer failed", e)
        is_this_correct = 0
    if is_this_correct == 0:
        task = row[0]
        final_answer = row[-2]
        prompt_input = {
                "question": task,
                "answer": final_answer
        }
        prompt = template.invoke(prompt_input)
        messages = [
                {"role": "user", "content": prompt.to_messages()[0].content}
        ]
        
        response = client.chat.completions.create(
            messages = messages,
            model = model
        )
        final_answer: str = response.choices[0].message.content
        if "Educated guess:" in final_answer:
            final_answer = final_answer.replace("Educated guess:", "").strip()
        try:
            is_this_correct = question_scorer(
                ground_truth=row[-1],
                model_answer=final_answer
            )
        except Exception as e:
            print("question_scorer failed", e)
            is_this_correct = 0
    else:
        final_answer = row[-2]
    eval_data.append(
        row + (final_answer, is_this_correct)
    )
    print("Correct" if is_this_correct == 1 else 'Incorrect', "GT:",row[-1], "Prediction:", row[-2])
    correct += is_this_correct
import pandas as pd

df = pd.DataFrame(eval_data, columns=["question", 'task_id', 'answer', 'ground_truth', 'formatted_answer', "correct"])
df.to_csv(args.data_path.replace('.db', '.csv'))
print("Total:", n_total)
print("Correct Item:", correct)
print("Accuracy:", round(100 * correct / n_total, 2), "%")