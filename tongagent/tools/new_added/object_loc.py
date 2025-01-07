from transformers import Tool
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class ObjectLOCTool(Tool):
    name = "objectlocation"
    description = "A tool that can localize objects in given images, outputing the bounding boxes of the objects."
    inputs = {
        "object": {"description": "the object that need to be localized", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to localize objects. This should be a local path to downloaded image.",
            "type": "string",
        },
    }
    output_type = "any"


    model_path = "google/owlvit-base-patch32"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path)
    model = model.to(device)


    def forward(self, object: str, image_path: str) -> list:
        image = Image.open(image_path)
        image = image.convert('RGB')

        texts=[]
        texts.append(f'a photo of {object}')
        texts=[texts]    

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs=inputs.to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        i = 0  
        text = texts[i]   
        output=[]

        for box, score, pred in zip(results[i]["boxes"], results[i]["scores"], results[i]["labels"]):
            # output.append(dict(score=score.item(), label=text[pred], box=[round(i, 2) for i in box.tolist()]))
            output.append([round(i, 2) for i in box.tolist()])

        return output
