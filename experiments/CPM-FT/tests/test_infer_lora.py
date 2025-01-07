import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from inference.utils import load_pretrained_model_lora
torch.manual_seed(0)

model, tokenizer = load_pretrained_model_lora("output/cpm_v2_6_7680255/")
# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('./assets/airplane.jpeg').convert('RGB')

# First round chat 
question = "Tell me the model of this aircraft."
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print("=" * 10)
print(answer)
print("=" * 10)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)