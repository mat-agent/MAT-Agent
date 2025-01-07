import argparse
import json
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
args = parser.parse_args()


files = os.listdir(args.data_path)

files = [os.path.join(args.data_path, f) for f in files]

files = [f for f in files if os.path.isdir(f)]

counts = []
for f in files:
    
    f = os.path.join(f, "agent_memory.json")
    with open(f, "r") as f:
        dataset = json.load(f)

    conv = dataset["conversations"]
    turn = len(conv)
    steps = (turn - 2) // 2
    print(steps)
    counts.append(steps)
    # print(conv)
    # break

import matplotlib.pyplot as plt

plt.figure(dpi=300)
plt.hist(counts, bins=7)
plt.xlabel("Steps")
plt.ylabel("Task counts")
plt.grid()
plt.show()