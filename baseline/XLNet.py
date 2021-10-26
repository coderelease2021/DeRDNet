import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from transformers import XLNetTokenizer, XLNetModel
import torch.nn as nn
'''
import torch
from transformers import XLNetTokenizer, BertTokenizer, DistilBertTokenizer
from baseline.unit import collate_fn
from baseline.DistilBERT import BaseModel
class Config(object):
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.bert_path = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = BaseModel
        self.collate_fn = collate_fn
        self.emb_dim = 256
        self.batch_size = batch_size
        self.warmup_lr = 1e-5
        self.data = 'release_data_sample.pkl'
        self.lr = 4e-5
        self.max_text_cut = 1024
        self.warmup_epoch = 1
        self.train_epoch = 10
        self.neg_test = True
config = Config(16)
'''
class DataEncoder(nn.Module):
    def __init__(self, config):
        super(DataEncoder, self).__init__()

        self.embedding = torch.nn.Embedding(23, config.emb_dim, padding_idx=0)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.activation(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.bert = XLNetModel.from_pretrained(config.bert_path)

    def forward(self, x):
        text = x[0]
        mask = x[1]
        outputs = self.bert(text, mask)
        return outputs[0][:, 0, :]




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.data_encoder = DataEncoder(config)
        self.trans_layer = nn.Linear(768, config.emb_dim)
        self.drop_layer = nn.Dropout(0.5)
        self.trans_activation = nn.Tanh()
    def forward(self, data, text):
        data_embedding = self.drop_layer(self.data_encoder(data))
        text_embedding = self.text_encoder(text)
        text_embedding = self.trans_layer(text_embedding)
        text_embedding = self.drop_layer(self.trans_activation(text_embedding))
        #total_embedding = torch.cat([data_embedding, text_embedding], 1)
        return data_embedding, text_embedding

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.encoder = Encoder(config)
        self.classify_layer = nn.Linear(2*config.emb_dim, 2)
        self.classify_layer_reason = nn.Linear(2 * config.emb_dim, 4)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)

    def forward(self, data, text):
        data_embedding, text_embedding = self.encoder(data, text)
        embedding = torch.cat([data_embedding, text_embedding], 1)
        prob_logit = self.classify_layer(embedding)
        prob_logit_reason = torch.sigmoid(self.classify_layer_reason(embedding))
        prob = self.output_activation(prob_logit)
        return prob_logit, prob_logit_reason, prob