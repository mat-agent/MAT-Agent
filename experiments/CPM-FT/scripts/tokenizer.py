
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)


print(tokenizer.decode([151646, 151647, 151656, 151657]))