import os

from transformers.agents import load_tool, Tool
from tongagent.utils import CACHE_FOLDER, gen_random_id
from diffusers import FluxPipeline
from diffusers import DiffusionPipeline

import torch

class ImageGenerationTool(Tool):
    description = "This is a tool that creates an image according to a prompt, which is a text description."
    name = "image_generator"
    inputs = {"prompt": {"type": "string", "description": "The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc."}}
    output_type = "any"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        if model_id == "black-forest-labs/FLUX.1-dev":
            # model_path = '/scratch/zhangbofei/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/'
            pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
            pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        else:
            raise ValueError(f"unk model {model_id}")
        self.pipeline = pipeline
        self.pipeline.to("cuda")
        self.model_id = model_id
        # pipeline.enable_model_cpu_offload()
        
    def forward(self, prompt):
        if self.model_id == "stabilityai/stable-diffusion-xl-base-1.0":
            image = self.pipeline(
                    prompt=prompt
                ).images[0]
        else:
            image = self.pipeline(
                prompt,
                height=512,
                width=512,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            
        output_image_path = os.path.join(CACHE_FOLDER, f"{gen_random_id()}.jpeg")
        image.save(output_image_path)
        # output_image.save(output_image_path)
        print("save to", output_image_path)
        return output_image_path

if __name__ == "__main__":
    tool = ImageGenerationTool()

    image_path = tool.forward("high-res, photorealistic street view")
