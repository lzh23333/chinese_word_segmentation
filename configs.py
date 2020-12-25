"""训练配置
"""
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertModel, BertTokenizer
from model import Configs, BertCWS


class PipelineConfigs:

    def __init__(self, linears=None):
        """模型配置
        Args:
            linears (list [int]): 分类器每层的feature数.
        """
        # model configs
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        # self.bert_model = BertModel.from_pretrained(
        #     "./models/bert-base-chinese",
        #     local_files_only=True
        # )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # self.tokenizer = BertTokenizer.from_pretrained(
        #     "./models/bert-base-chinese",
        #     local_files_only=True
        # )
        self.classifier_configs = Configs()
        if linears is not None:
            self.classifier_configs.linears = linears

        self.model = BertCWS(self.bert_model, self.classifier_configs)

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


class TestConfig:

    def __init__(self):
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.classifier_configs = Configs()

        self.classifier_configs.linears = [768, 256, 4]
        self.model = BertCWS(self.bert_model, self.classifier_configs)
        self.model.load_state_dict(torch.load("./models/bertCWS-256-4.pt"))
        self.max_len = 300
        self.batch_size=2
        self.test_filename = "./data/test.id.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConfig1:

    def __init__(self):
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.classifier_configs = Configs()

        self.classifier_configs.linears = [768, 4]
        self.model = BertCWS(self.bert_model, self.classifier_configs)
        self.model.load_state_dict(torch.load("./models/bertCWS-4.pt"))
        self.max_len = 300
        self.batch_size=2
        self.test_filename = "./data/test.id.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")