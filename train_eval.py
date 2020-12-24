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
import numpy as np


def collate_fn(x):
    inputs = [a[0] for a in x]
    labels = [a[1] for a in x]
    max_len = max([len(x) for x in inputs])
    attention_mask = [([1] * len(a)) + ([0] * (max_len - len(a))) for a in inputs]
    inputs = [a + ([0] * (max_len - len(a))) for a in inputs]
    inputs = torch.LongTensor(inputs)
    attention_mask = torch.IntTensor(attention_mask)
    return inputs, labels, attention_mask


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
        labels[i] = labels[i][:len_i-2]
    outputs = torch.cat(preds)
    labels = torch.LongTensor(reduce(lambda a, b: a + b, labels))
    return outputs, labels


def train(model, dataloader, criterion, optimizer,
          device, writer=None, T=50, clip=1):
    """训练代码，所有相关配置在函数外配置完成

    Args:

    Returns:
        loss (float): 平均loss.
    """
    model.to(device)
    model.train()
    total_loss = 0
    recent_loss = []
    for i, batch in tqdm(enumerate(dataloader), desc="training"):
        sentences, labels, attention_mask = batch
        outputs = model(sentences.to(device),
                        attention_mask.to(device))
        outputs, labels = output_resize(outputs, labels,
                                        attention_mask)
        loss = criterion(outputs, labels.to(device))
        recent_loss.append(loss.item())
        total_loss += loss.item()
        if i % T == 0 and recent_loss != [] and writer is not None:
            writer.add_scalar("Loss/Train", np.mean(recent_loss), i // T)
            recent_loss = []
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del loss, outputs

    return total_loss / len(dataloader)


def evaluation(model, dataloader, criterion,
               device, writer=None, T=50):
    """验证集测试代码，所有相关配置在函数外配置完成

    Returns:
        loss (float): 平均loss.
    """
    model.to(device)
    model.eval()
    eval_loss = 0
    recent_loss = []
    for i, batch in tqdm(enumerate(dataloader), desc="eval"):
        sentences, labels, attention_mask = batch
        outputs = model(sentences.to(device),
                        attention_mask.to(device))
        outputs, labels = output_resize(outputs, labels,
                                        attention_mask)
        loss = criterion(outputs, labels.to(device))
        recent_loss.append(loss.item())
        if i % T == 0 and recent_loss != [] and writer is not None:
            writer.add_scalar("Loss/Valid", np.mean(recent_loss), i // T)
            recent_loss = []
        eval_loss += loss.item()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return eval_loss / len(dataloader)


def test(model, testloader, device):
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
        sentences, labels, attention_mask = batch
        outputs = model(sentences.to(device),
                        attention_mask.to(device))
        outputs, _ = output_resize(outputs, labels,
                                   attention_mask)
        outputs = torch.argmax(outputs, dim=1).tolist()
        preds += outputs
        trues += reduce(lambda a, b: a + b, labels)
    return preds, trues
