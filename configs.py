"""训练配置
"""
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertModel, BertTokenizer
from model import Configs, BertCHW


class PipelineConfigs:

    def __init__(self):
        # model configs
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.classifier_configs = Configs()
        self.model = BertCHW(self.bert_model, self.classifier_configs)
        # train configs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {"params": self.model.bert_model.parameters(), "lr": 1e-5},
            {"params": self.model.classifier.parameters(), "lr": 1e-2}
        ])
        self.batch_size = 3
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.epochs = 4
        self.train_filename = "./data/train.id.json"
        self.test_filename = "./data/test.id.json"
        self.ratio = 0.9    # 训练集比例
        self.dst = "./model.pt"
        self.max_len = 300  # 句子的最大长度，太长的会被裁剪
