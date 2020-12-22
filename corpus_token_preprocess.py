#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   corpus_token_preprocess.py
@Time    :   2020/12/22 14:53:54
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   利用tokenizer对corpus进行进一步的预处理，保存(id, label)对，
             防止tokenizer导致的token数与label数不一致情况.
'''
import json
from transformers import BertTokenizer
from tqdm import tqdm
from utils import BEGIN, MID, END, SINGLE


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
corpus_files = ["./data/train.json", "./data/test.json"]
dst = ["./data/train.id.json", "./data/test.id.json"]
for n, filename in enumerate(corpus_files):
    print(f"processing {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    updates = []
    for tokens, labels in tqdm(data):
        update_ids = []
        update_labels = []
        for i, token in enumerate(tokens):
            ids = tokenizer([token])["input_ids"][0]
            if len(ids) > 3:
                # update_tokens += tokenizer.convert_ids_to_tokens(ids[1:-1])
                update_ids += ids[1:-1]
                update_labels += ([BEGIN] + [MID] * (len(ids) - 2) + [END])
            else:
                # update_tokens.append(token)
                update_ids += ids[1:-1]
                update_labels.append(labels[i])
            # update_tokens.append(token)
        updates.append((update_ids, update_labels))
    with open(dst[n], "w", encoding="utf-8") as f:
        f.write(json.dumps(updates, ensure_ascii=False))
