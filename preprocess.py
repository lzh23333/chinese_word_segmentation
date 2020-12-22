#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2020/12/18 17:00:58
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   读取所有语料并进行预处理以及标记，最后存到json文件中.
'''

from tqdm import tqdm
import json
from transformers import BertTokenizer
from utils import text_process, text_tagging


train_corpus_labels = []
test_corpus_labels = []
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
datasets = ["as", "cityu", "msr", "pku"]
is_traditions = [True, True, False, False]
for i in range(4):
    train_corpus = []
    test_corpus = []
    d = datasets[i]
    is_tradition = is_traditions[i]
    with open(f"./data/icwb2-data/training/{d}_training.utf8", "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc=f"reading train {d}"):
            train_corpus.append(line[:-1])
    with open(f"./data/icwb2-data/gold/{d}_test_gold.utf8", "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc=f"reading test {d}"):
            test_corpus.append(line[:-1])
    for text in tqdm(train_corpus, desc="tagging"):
        text = text_process(text, is_tradition)
        char_list, labels = text_tagging(text)
        train_corpus_labels.append([char_list, labels])
    for text in tqdm(test_corpus, desc="tagging"):
        text = text_process(text, is_tradition)
        char_list, labels = text_tagging(text)
        test_corpus_labels.append([char_list, labels])


with open("./data/train.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_corpus_labels, ensure_ascii=False))
with open("./data/test.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(test_corpus_labels, ensure_ascii=False))
