import sys
sys.path.insert(0, "./")
import sqlite3

from tongagent.agents.data_sampling_agent import create_agent
from tongagent.utils import load_config
from datasets import load_dataset
import os
import argparse
from typing import Optional
from tqdm import tqdm

DATA_NAME = "2023_level2"
SPLIT = "validation"

def run(agent, raw_question, attachment_name):
    if attachment_name is not None and attachment_name.strip() != "":
        question = f"{raw_question}\nAttachment: data/GAIA/2023/{SPLIT}/{attachment_name}"
    else:
        question = raw_question
    
    if attachment_name is not None and (attachment_name.endswith(".png") or attachment_name.endswith(".jpg")):
        agent.image_paths = [f'data/GAIA/2023/{SPLIT}/{attachment_name}']
    else:
        agent.image_paths = []
        
    result = agent.run(question)
    agent.save_trajectory()
    return result


ds = load_dataset("gaia-benchmark/GAIA", DATA_NAME, split=SPLIT)
agent = create_agent(llm_engine="tonggpt", task="gaia", error_tolerance=3)

# selected = "e8cb5b03-41e0-4086-99e5-f6806cd97211"
# item = [item for item in ds if item["task_id"] == selected][0]
# print("item", item)

# question = "The object in the British Museum's collection with a museum number of 2012,5015.17 is the shell of a particular mollusk species. According to the abstract of a research article published in Science Advances in 2021, beads made from the shells of this species were found that are at least how many thousands of years old?"

# question = "The year is 2022. I am at the National Air and Space Museum east of the Potomac River. I want to go to Fire Station 301 DCA ARFF using the metro. I go in the wrong direction and end up at the station closest to Cleveland Elementary School. How many metro stations am I away from my original destination if I don't change lines? Your answer should be a numerical integer value."

# question = "In the YouTube 360 VR video from March 2018 narrated by the voice actor of Lord of the Rings' Gollum, what number was mentioned by the narrator directly after dinosaurs were first shown in the video?"

# question = "In the YouTube 360 VR video from March 2018 narrated by the voice actor of Lord of the Rings' Gollum, what chemical terminology was mentioned by the narrator directly after H2O were first mentioned in the video?"

question = "Visit Bofei's Site to find his current position in industry."
file_name = None
result = run(
    agent,
    raw_question=question,
    attachment_name=file_name
)

print(result)