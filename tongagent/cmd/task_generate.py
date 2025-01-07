import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    
    args = parser.parse_args()
    with open(args.data_path, "r") as f:
        dataset = json.load(f)

    conv = dataset["conversations"]
    task = conv[1]["content"]
    if not task.endswith("\n"):
        task += "\n"
        
    for i in range(2, len(conv)):
        content = conv[i]["content"]
        if i % 2 == 0:
            if not content.startswith("Thought:"):
                raise ValueError("This trajectory is malformed")
            
            task += "\n"
            task += content
            
        else:
            if not content.startswith("[OUTPUT OF STEP") or "Observation:" not in content:
                raise ValueError("This trajectory is malformed")

            content_idx = content.find("]")
            task += "\n"
            task += content[content_idx+1:].strip()
        if not task.endswith("\n"):
            task += "\n"
    print(task)
if __name__ == "__main__":
    main()