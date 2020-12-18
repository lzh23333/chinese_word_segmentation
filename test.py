# %%
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# %%
inputs = tokenizer(["你好，爱丽丝", "你是谁"], return_tensors="pt", padding=True)
outputs = model(**inputs)
print(outputs[0].size())
last_hidden_state = outputs.last_hidden_state
# %%
