import datasets
from datasets import load_dataset

def convert(item):
    pass
ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d")
dataset = ds["train"]
for item in dataset:
    print(item)
    break