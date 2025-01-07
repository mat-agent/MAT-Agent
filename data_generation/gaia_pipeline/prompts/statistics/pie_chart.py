import matplotlib.pyplot as plt
import numpy as np

import json

# Data to plot
def pie_chart(labels, sizes, pdf_save_path):


    # Generate a list of colors from a colormap
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.linspace(0, 1, len(labels)))

    # Plot
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=140, textprops={'fontsize': 14},
            pctdistance=0.9)  # Move the percentage text outward

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Programming Language Usage', fontsize=20)

    # Save the figure as a PDF
    plt.savefig(pdf_save_path)

    plt.show()



def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    return data

def write_json(data, filename):
    """
    Write a JSON-compatible Python dictionary to a file.

    :param data: The JSON-compatible dictionary to write.
    :param filename: The name of the file to write to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")



json_path="data/final_dataset/tool_statistics.json"
pdf_save_path='data/final_dataset/tool_statistics.pdf'

# json_path="data/final_dataset/file_statistics.json"
# pdf_save_path='data/final_dataset/file_statistics.pdf'

# json_path="data/final_dataset/topic_statistics.json"
# pdf_save_path='data/final_dataset/topic_statistics.pdf'

json_data=load_json(json_path)
labels=list(json_data.keys())
values=list(json_data.values())

pie_chart(labels,values,pdf_save_path)