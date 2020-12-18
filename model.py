#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/12/18 14:23:20
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   Chinese Word Segmentation With Bert
'''
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


class Configs:

    def __init__(self):
        self.dropout = 0.5
        self.activate = nn.ReLU()
        self.linears = [768, 256, 4]


class BertCHW(nn.Module):
    """ Bert for Chinese Word Segmentation.
    """
    def __init__(self, bert_model: BertModel, configs: Configs):
        self.bert_model = bert_model
        self.dropout = nn.Dropout(configs.dropout)
        self.activate_layer = configs.activate
        self.classifier = nn.ModuleList()
        for i in range(len(configs.linears) - 1):
            in_dim = configs.linears[i]
            out_dim = configs.linears[i+1]
            self.classifier.append(
                nn.Linear(in_dim, out_dim)
            )

    def forward(
        self,
        input_ids,
        attention_mask=None
    ):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]    # [batch_size, sequence len, 768]
        x = self.dropout(sequence_output)
        for fc in self.classifier[:-1]:
            x = self.activate_layer(fc(x))
        x = self.fc(x)
        # [batch size, sequence len, 4]
        return x

