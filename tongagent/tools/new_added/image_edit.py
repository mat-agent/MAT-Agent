from transformers import Tool
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tongagent.utils import CACHE_FOLDER, gen_random_id
import torch
import os

class ModelSingleton():
    def __new__(cls):
        if hasattr(cls, "pipe"):
            return cls
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        cls.pipe = pipe
        return cls
        
class ImageEditTool(Tool):
    name = "image_edit"
    description = "A tool that can edit image based on the user prompt. Return a file path for printing."
    inputs = {
        "prompt": {
            "description": "The user prompt that instruct how to edit the image.",
            "type": "string",
        },
        "image_path": {
            "description": "The image path that this tool will try to edit.",
            "type": "string",
        },
    }
    output_type = "string"
    
    
    def forward(self, prompt: str, image_path: str) -> str:
        print("ImageEditTool input", prompt, image_path)
        image = Image.open(image_path).convert("RGB")        
        images = ModelSingleton().pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
        output_image = images[0]
        output_image_path = os.path.join(CACHE_FOLDER, f"{gen_random_id()}.png")
        output_image.save(output_image_path)
        print("save to", output_image_path)
        return output_image_path