"""训练配置
"""
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertModel, BertTokenizer
from model import Configs, BertCHW


class PipelineConfigs:

    def __init__(self):
        # model configs
        # self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.bert_model = BertModel.from_pretrained(
            "./models/bert-base-chinese",
            local_files_only=True
        )
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(
            "./models/bert-base-chinese",
            local_files_only=True
        )
        self.classifier_configs = Configs()
        self.classifier_configs.linears = [768, 4]

        self.model = BertCHW(self.bert_model, self.classifier_configs)

        # train configs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {"params": self.model.bert_model.parameters(), "lr": 1e-5},
            {"params": self.model.classifier.parameters()}],
            lr=1e-2
        )
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.epochs = 8
        self.train_filename = "./data/train.id.json"
        self.test_filename = "./data/test.id.json"
        self.ratio = 0.9    # 训练集比例
        self.dst = f"./model-{datetime.now()}.pt"
        self.max_len = 300  # 句子的最大长度，太长的会被裁剪
