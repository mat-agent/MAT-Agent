from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print(tokenizer.pad_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.bos_token_id)
print(tokenizer.encode("<|im_start|>"))
print(tokenizer.encode("<|im_end|>"))
print(tokenizer.encode("<|im_start|>assistant"))
print(tokenizer.encode("<|im_end|>"))

print(tokenizer.decode([872]))

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I am fine, thank you!"},
]

print(tokenizer.apply_chat_template(msgs, tokenize=False))