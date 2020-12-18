#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/12/18 14:58:16
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   including Dataset, data preprocess and
             chinese segmentation demo function
'''
from torch.utils.data import Dataset
from tqdm import tqdm
import re
from hanziconv import HanziConv


# define 
BEGIN = 0
END = 1
MID = 2
SINGLE = 3

def full2half(s):
    """全角转为半角

    Args:
        s (str)
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


def text_process(text, is_tradition):
    """语料预处理

    Args:
        text (str): 分词句子.
        is_tradition (bool): 是否为繁体语料.
    Returns:
        text (str): 预处理后的语料.
    """
    text = full2half(text).lower()
    if is_tradition:
        text = HanziConv.toSimplified(text)
    return text


def text_tagging(text):
    """对中文句子进行标注

    Args:
        text (str): 中文句子.
    Returns:
        char_list (list [str]): 字符序列（数字、单词不切分）.
        labels (list [int]): 标注序列.
    """
    char_list = []
    labels = []
    for word in text.split(" "):
        if word not in ["", "\t", "\n"]:
            cl, ls = chinese_word_tagging(word)
            char_list += cl
            labels += ls
    return char_list, labels


def chinese_word_tagging(word):
    """对语句序列进行标注.

    Args:
        word (str): 中文词，可能包含多个字符.
    Returns:
        char_list (list [str]): 字符序列（数字、单词不切分）.
        labels (list [int]): 标注序列.
    """
    char_list = seg_char(word)
    labels = [MID for _ in char_list]
    if len(labels) == 1:
        labels = [SINGLE]
    else:
        labels[0] = BEGIN
        labels[-1] = END
    return char_list, labels


def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    result = []
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    for p in parts:
        chars = pattern.split(p)
        chars = [w for w in chars if len(w.strip())>0]
        result += chars
    return result