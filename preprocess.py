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
from utils import text_process, text_tagging


corpus_labels = []
datasets = ["as", "cityu", "msr", "pku"]
is_traditions = [True, True, False, False]
for i in range(4):
    corpus = []
    d = datasets[i]
    is_tradition = is_traditions[i]
    with open(f"./data/icwb2-data/training/{d}_training.utf8", "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc=f"reading dataset {d}"):
            corpus.append(line[:-1])
    for text in tqdm(corpus, desc="tagging"):
        text = text_process(text, is_tradition)
        char_list, labels = text_tagging(text)
        corpus_labels.append([char_list, labels])


with open("./data/total.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(corpus_labels, ensure_ascii=False))
