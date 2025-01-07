from inference.utils import load_pretrained_model
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from loguru import logger
import copy

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_paths, gts):
        self.questions = questions
        self.image_paths = image_paths
        self.gts = gts
       

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        return image, self.questions[index], self.gts[index], self.image_paths[index]
    

    def __len__(self):
        return len(self.questions)

def main():
    with open("data/FeedbackReflection/json/test/FIRE-test-Student.json", "r") as f:
        dataset = json.load(f)
    questions = []
    gts = []
    images = []
    for item in dataset:
        question = item["raw"]["question"]["value"]
        image = os.path.join("data/FIRE-test/", item["image"])
        if "mathverse" in image:
            image = image.replace("mathverse", "mathverse/images")
        Image.open(image)
        gt = item["raw"]["groundtruth"]["value"]
        questions.append(question)
        gts.append(gt)
        images.append(image)
        
    model, tokenizer = load_pretrained_model()
    dataset = CustomDataset(
        questions,
        images,
        gts,
    )
    outputs = []
    for item in tqdm(dataset):
        image, question, gt, image_path = item
        msgs = [{'role': 'user', 'content': [image, question]}]
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        output_item = {
            "question": question,
            "groundtruth": gt,
            "image_path": image_path,
            "answer": answer
        }
        outputs.append(output_item)
        if len(outputs) % 10 == 0:
            with open("output/cpm_fire_test.json", "w") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
                
        print("Q", question)
        print("A", answer)
        print("GT", gt)

    with open("output/cpm_fire_test.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()