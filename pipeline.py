#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pipeline.py
@Time    :   2020/12/22 09:58:00
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   训练和测试
'''
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from configs import PipelineConfigs
from utils import DatasetCHW
from train_eval import train, evaluation, collate_fn, test


def main():
    configs = PipelineConfigs()
    writer = SummaryWriter()

    train_set = DatasetCHW(configs.train_filename)
    test_set = DatasetCHW(configs.test_filename)
    tokenizer = configs.tokenizer

    L = len(train_set)
    train_num = int(L * configs.ratio)
    train_set, val_set = random_split(train_set, [train_num, L - train_num])
    # train and eval
    train_loader = DataLoader(
        train_set,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    for _ in range(configs.epochs):
        train(
            configs.model,
            train_loader,
            configs.criterion,
            configs.optimizer,
            tokenizer,
            configs.device,
            writer=writer
        )
        evaluation(
            configs.model,
            val_loader,
            configs.criterion,
            tokenizer,
            configs.device,
            writer=writer
        )
    torch.save(configs.model.state_dict(), configs.dst)

    # test
    preds, trues = test(
        configs.model,
        test_loader,
        tokenizer,
        configs.device
    )
    print(classification_report(trues, preds))


if __name__ == "__main__":
    main()
