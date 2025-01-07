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



class SearchTool(Tool):
    name = "ask_search_agent"
    description = """This will send a message to a team member that will browse the internet to answer your question. Ask him for all your web-search related questions, but he's unable to do problem-solving. Provide him as much context as possible, in particular if you need to search on a specific timeframe! And don't hesitate to provide them with a complex search task, like finding a difference between two webpages."""

    inputs = {
        "query": {
            "description": "Your question, as a natural language sentence with a verb! You are talking to an human, so provide them with as much context as possible! DO NOT ASK a google-like query like 'paper about fish species 2011': instead ask a real sentence like: 'What appears on the last figure of a paper about fish species published in 2011?'",
            "type": "text",
        }
    }
    output_type = "text"

    
    config = load_config()
    API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
    REGION = config.visualizer.region
    # ENDPOINT = f"{API_BASE}/{REGION}"
    from openai import OpenAI
    # client = OpenAI(api_key="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")
    client = AzureOpenAI(
        api_key=config.visualizer.api_key,
        api_version="2024-02-01",
        azure_endpoint=f"{API_BASE}/{REGION}",
    )

    # client = AzureOpenAI(
    #     api_key=config.visualizer.api_key,
    #     api_version="2024-02-01",
    #     azure_endpoint=f"{API_BASE}/{REGION}",
    # )

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




    def forward(self, query: str) -> str:
        add_note = False
        question=query
            
        messages=[
            {
                'role': 'system',
                'content': 'You need to act as a web search agent. Given a query, you need to simulate searching for the query on the Internet, and output the search results.'
            },
            {
                'role': 'user',
                'content': [
                    {
                        "type": "text",
                        "text": question
                    },
                ]
            }
        ]


        response = self.get_chat_response(messages=messages, model=self.config.visualizer.model_name, temperature=1, max_tokens=500)
        output=response

        # try:
        #     output = response.json()['choices'][0]['message']['content']
        # except Exception:
        #     raise Exception(f"Response format unexpected: {response.json()}")

        answer = "Search Result:\n"
        answer += str(response)
        # print("SearchTool output:\n", answer)
        return answer