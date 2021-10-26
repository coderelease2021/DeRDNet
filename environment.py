import torch
from transformers import DistilBertTokenizer
from model_folder.dynamic import DynamicDecisionFilterGate as modelc
class Config(object):
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # è®¾å¤‡
        self.bert_path = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = modelc
        self.pad_replace = False
        self.filter_sizes = [3]
        self.gpuid = 3
        self.reason_tokenizer = None
        self.reason_vocab_size = None
        self.emb_dim = 256
        self.reason_length = 25
        self.batch_size = batch_size
        self.warmup_lr = 1e-6
        self.data = 'release_data_sample.pkl'
        self.lr = 2e-6
        self.max_text_cut = 512
        self.warmup_epoch = 2
        self.train_epoch = 20
        self.reason = False
        self.lam = 0.75
        self.use_simple_reason = True
        self.d_emb_dim = 128
        self.d_step = 3
        self.d_bn = True
        self.d_method = 'avg'
        self.neg_test = True
        self.max_attemp = 3
config = Config(16)
