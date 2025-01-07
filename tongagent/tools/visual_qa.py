from PIL import Image
import base64
from io import BytesIO
import json
import os
import requests
from typing import Optional
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, Tool
import uuid
import mimetypes
from openai import AzureOpenAI
import time

from tongagent.utils import load_config
# from dotenv import load_dotenv

# load_dotenv(override=True)

# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"
    
        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}


import base64
import requests
from PIL import Image
from io import BytesIO

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path



class VisualQAGPT4Tool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
    }
    output_type = "string"

    
    config = load_config()
    API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
    if config.visualizer.model_name.startswith("Qwen"):
        from tongagent.llm_engine.qwen import QwenEngine
        client = QwenEngine(config.visualizer.model_name)
    else:
        REGION = config.visualizer.region
        client = AzureOpenAI(
            api_key=config.visualizer.api_key,
            api_version="2024-02-01",
            azure_endpoint=f"{API_BASE}/{REGION}",
        )

    def get_chat_response(self, messages, model="gpt-4o-mini-2024-07-18", temperature=1, max_tokens=2048, n=1, patience=10, sleep_time=2):
        while patience > 0:
            patience -= 1
            try:
                # print(messages[0])
                # print(" CONNECTING TO GPT4-V")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n
                )
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0]:
                        return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                pass
        return ""



    def forward_qwen(self, question: Optional[str], image_path: str) -> str:
        add_note = False
        if type(question) is not str:
            raise Exception("parameter question should be a string.")
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        
        image_paths = []
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            print ('The type of input image is ', type(image_path))
            raise Exception("The type of input image should be string (image path)")

        messages = [
            {"role": "user", "content": question}
        ]
        output = self.client.call_vlm(
            messages,
            image_paths = image_paths
        )

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output
    
    def forward_gpt(self, question: Optional[str], image_path: str) -> str:
        add_note = False
        if type(question) is not str:
            raise Exception("parameter question should be a string.")
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
            
        if isinstance(image_path, str):
            base64_image = encode_image(image_path)
        elif image_path is None:
            raise Exception("Image_path is None, please provide a valid image path")
        else:
            print('The type of input image is ', type(image_path))
            raise Exception("The type of input image should be string (image path)")

            
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        "type": "text",
                        "text": question
                    },
                    {   
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]


        response = self.get_chat_response(messages=messages, model=self.config.visualizer.model_name, temperature=1, max_tokens=500)
        output=response

        # try:
        #     output = response.json()['choices'][0]['message']['content']
        # except Exception:
        #     raise Exception(f"Response format unexpected: {response.json()}")

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output
    
    def forward(self, question: Optional[str], image_path: str) -> str:
        if self.config.visualizer.model_name.startswith("Qwen"):
            return self.forward_qwen(question, image_path)
        else:
            return self.forward_gpt(question, image_path)