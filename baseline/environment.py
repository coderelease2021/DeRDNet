import torch
from transformers import XLNetTokenizer, BertTokenizer, DistilBertTokenizer
from baseline.unit import collate_fn_han
from baseline.Han import BaseModel
class Config(object):
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.bert_path = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = BaseModel
        self.collate_fn = collate_fn_han
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