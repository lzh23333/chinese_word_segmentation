# %%
from os.path import split
from transformers import BertTokenizer, BertModel
from model import BertCHW, Configs
import torch

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# %%
sentence = ["你 是 谁".split(" "), "你 好".split(" ")]
print(sentence)
# %%
CHW_model = BertCHW(model, Configs())
print(CHW_model)
# %%
inputs = tokenizer(sentence, return_tensors="pt", padding=True, is_split_into_words=True)
# %%
y = CHW_model(inputs["input_ids"], inputs["attention_mask"])
print(y)
# %%
x = torch.randn(10, 4).to(torch.device("cuda"))
y = torch.argmax(x, dim=1)
y.size()
# %%
y = y.tolist()
type(y[0])
# %%
