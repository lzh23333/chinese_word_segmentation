# %%
from os.path import split
from transformers import BertTokenizer, BertModel
from model import BertCHW, Configs
import torch
from train_eval import output_resize

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# %%
sentence = [['fa', '##t', ')', '、', '及', 'ibm', '##bi', '##o', '.']]
print(sentence)
# %%
CHW_model = BertCHW(model, Configs())
print(CHW_model)
# %%
inputs = tokenizer(sentence, padding=True, is_split_into_words=True)
# %%
print(inputs)
tokens2 = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(tokens2)

# %%

# %%
