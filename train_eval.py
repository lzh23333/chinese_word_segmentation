#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_eval.py
@Time    :   2020/12/18 14:54:39
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   train eval utils function
'''
# %%
import enum
import torch
from tqdm import tqdm
from functools import reduce


def collate_fn(x):
    inputs = [a[0] for a in x]
    labels = [a[1] for a in x]
    return inputs, labels


def output_resize(outputs, labels, attention_mask):
    """将有padding的输出位剔除，并转为二维tensor.

    Args:
        outputs (Tensor): [batch_size, length, class_num].
        labels (list [list [int]]): 每个样本对应的label序列.
        attention_mask (Tensor): [batch_size, length].
    Returns:
        outputs (Tensor): [L, class_num].
        labels (LongTensor): [L].
    """
    preds = []
    for i in range(len(labels)):
        len_i = torch.sum(attention_mask[i])
        p = outputs[i, 1: len_i-1, :]
        preds.append(p)
    outputs = torch.cat(preds)
    labels = torch.LongTensor(reduce(lambda a, b: a + b, labels))
    return outputs, labels


def train(model, dataloader, criterion, optimizer,
          tokenizer, device, max_len=300, writer=None):
    """训练代码，所有相关配置在函数外配置完成

    Args:

    Returns:
        loss (float): 平均loss.
    """
    model.to(device)
    model.train()
    total_loss = 0

    for i, batch in tqdm(enumerate(dataloader), desc="training"):
        inputs, labels = batch
        inputs = tokenizer(inputs, return_tensors="pt", max_length=max_len,
                           padding=True, is_split_into_words=True)
        outputs = model(inputs["input_ids"].to(device),
                        inputs["attention_mask"].to(device))
        outputs, labels = output_resize(outputs, labels,
                                        inputs["attention_mask"])
        print(outputs.size(), labels.size())
        loss = criterion(outputs, labels.to(device))
        total_loss += loss.item()
        if i % 20 == 0 and writer is not None:
            writer.add_scalar("Loss/Train", loss.item(), i // 20)
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def evaluation(model, dataloader, criterion, tokenizer,
               device, max_len=300, writer=None):
    """验证集测试代码，所有相关配置在函数外配置完成

    Returns:
        loss (float): 平均loss.
    """
    model.to(device)
    model.eval()
    eval_loss = 0

    for i, batch in tqdm(enumerate(dataloader), desc="eval"):
        inputs, labels = batch
        inputs = tokenizer(inputs, return_tensors="pt", max_length=max_len,
                           padding=True, is_split_into_words=True)
        outputs = model(
            inputs["input_ids"].to(device),
            inputs["attention_mask"].to(device)
        )
        outputs, labels = output_resize(outputs, labels,
                                        inputs["attention_mask"])
        loss = criterion(outputs, labels.to(device))
        if i % 20 == 0 and writer is not None:
            writer.add_scalar("Loss/Valid", loss.item(), i // 20)
        eval_loss += loss.item()

    return eval_loss / len(dataloader)


def test(model, testloader, tokenizer, device, max_len=300,):
    """准确率测试

    Returns:
        preds (list [int]).
        trues (list [int]).
    """
    model.to(device)
    model.eval()

    preds = []
    trues = []
    for i, batch in tqdm(enumerate(testloader), desc="testing"):
        inputs, labels = batch
        inputs = tokenizer(inputs, return_tensors="pt",
                           max_length=max_len,
                           padding=True, is_split_into_words=True)
        outputs = model(
            inputs["input_ids"].to(device),
            inputs["attention_mask"].to(device)
        )
        outputs, _ = output_resize(outputs, labels, inputs["attention_mask"])
        outputs = torch.argmax(outputs, dim=1).tolist()
        preds += outputs
        trues += reduce(lambda a, b: a + b, labels)
    return preds, trues
