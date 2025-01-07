import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def load_pretrained_model():
    torch.manual_seed(0)

    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    return model, tokenizer

from peft import PeftModel

def load_pretrained_model_lora(peft_model_id):
    # model_id = 'openbmb/MiniCPM-V-2_6'
    model, tokenizer = load_pretrained_model()
    print("Load Lora")
    model = PeftModel.from_pretrained(model, peft_model_id)
    print("Lora merge and unload")
    model.merge_and_unload()
    return model, tokenizer