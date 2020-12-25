#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2020/12/25 09:26:40
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   测试模型性能
'''
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from configs import TestConfig, TestConfig1
from utils import DatasetCWS
from train_eval import collate_fn, test


def main():
    configs = TestConfig1()

    attrs = vars(configs)
    for k, v in attrs.items():
        print(k, v)
    test_set = DatasetCWS(configs.test_filename, max_len=configs.max_len)
    test_loader = DataLoader(
        test_set,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    preds, trues = test(
        configs.model,
        test_loader,
        configs.device
    )
    print(classification_report(trues, preds))


if __name__ == "__main__":
    main()