import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from transformers import XLNetTokenizer, XLNetModel
import torch.nn as nn
from math import floor
'''
import torch
from transformers import XLNetTokenizer, BertTokenizer, DistilBertTokenizer
from baseline.unit import collate_fn
from baseline.MultiResCNN import BaseModel
class Config(object):
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.bert_path = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = BaseModel
        self.collate_fn = collate_fn
        self.emb_dim = 256
        self.batch_size = batch_size
        self.warmup_lr = 1e-5
        self.data = 'release_data_sample.pkl'
        self.lr = 1e-4
        self.max_text_cut = 512
        self.warmup_epoch = 1
        self.train_epoch = 10
        self.neg_test = True
config = Config(16)
'''
class WordRep(nn.Module):
    def __init__(self, args, word_len):
        super(WordRep, self).__init__()

        self.gpu = True
        num_filter_maps = 50
        # add 2 to include UNK and PAD
        self.embed = nn.Embedding(word_len, args.emb_dim, padding_idx=0)
        self.feature_size = self.embed.embedding_dim


        self.embed_drop = nn.Dropout(p=0.2)

        self.conv_dict = {1: [self.feature_size, num_filter_maps],
                     2: [self.feature_size, 100, num_filter_maps],
                     3: [self.feature_size, 150, 100, num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, num_filter_maps]
                     }


    def forward(self, x):

        features = [self.embed(x)]

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x

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

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class OutputLayer(nn.Module):
    def __init__(self, args, Y, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y

class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.data_encoder = DataEncoder(args)
        self.word_rep = WordRep(args, 30522)
        self.conv = nn.ModuleList()
        self.drop_layer = nn.Dropout(0.2)
        self.classify_layer = nn.Linear(args.emb_dim, 2)
        self.classify_layer_reason = nn.Linear(args.emb_dim, 4)
        filter_sizes = "3,5,9,15,19,25".split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[1]
            for idx in range(1):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    0.2)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, 2, self.filter_num * 50)
        self.output_layer_reason = OutputLayer(args, 4, self.filter_num * 50)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)


    def forward(self, data, x):

        data_embedding = self.drop_layer(self.data_encoder(data))
        prob_logit = self.classify_layer(data_embedding)
        prob_logit_reason = self.classify_layer_reason(data_embedding)
        x = x[0]
        x = self.word_rep(x)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        prob_logit += self.output_layer(x)
        prob_logit_reason += self.output_layer_reason(x)
        prob_logit_reason = torch.sigmoid(prob_logit_reason)
        prob = self.output_activation(prob_logit)
        return prob_logit, prob_logit_reason, prob

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False






