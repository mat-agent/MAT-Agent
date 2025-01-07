from openai import OpenAI
import openai
import httpx
# vllm serve Qwen/Qwen2.5-7B-Instruct > output/vllm.log 2>&1 &
# vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2 > output/vllm.log 2>&1 &
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
import time
for i in range(60):
    try:
        print(client.models.list())
    except httpx.ConnectError as e:
        print("service might not ready!")
        time.sleep(10)
    except openai.APIConnectionError as e:
        print("service might not ready!")
        time.sleep(10)
    except Exception as e:
        raise e
    
elapse_times = []
while True:
    current = time.time()
    try:
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": "Hi"
                }],
            temperature=0.7,
            top_p=0.8,
            max_tokens=1,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
    except httpx.ConnectError as e:
        print("service might not ready!")
        time.sleep(60)
    except openai.APIConnectionError as e:
        print("service might not ready!")
        time.sleep(60)
    except Exception as e:
        raise e
        
    

    print("Chat response:", chat_response.choices[0].message.content)
    print("Elapse", round(time.time() - current, 2))
    elapse_times.append(round(time.time() - current, 2))
    time.sleep(600)
    
