import torch
import torch.nn as nn

from torch.nn.init import uniform_
'''
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
'''
class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        word_num_hidden = 128
        words_num = 30522
        words_dim = config.emb_dim
        self.mode = 'rand'
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim)
            rand_embed_init = uniform_(rand_embed_init, -0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x

class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = 128
        word_num_hidden = 128
        target_class = 11
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        #self.fc = nn.Linear(2 * sentence_num_hidden , target_class)
        self.soft_sent = nn.Softmax()

    def forward(self,x):
        sentence_h,_ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1,0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0)
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


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.data_encoder = DataEncoder(config)
        self.classify_layer = nn.Linear(256+256, 2)
        self.classify_layer_reason = nn.Linear(256+256, 4)
        self.output_activation = nn.Softmax(dim=-1)
        self.drop_layer = nn.Dropout(0.5)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)

    def forward(self, data, x):
        x = x[0]
        x = x.permute(1, 2, 0)  # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        x = self.sentence_attention_rnn(word_attentions)
        data_embedding = self.drop_layer(self.data_encoder(data))
        embedding = torch.cat([data_embedding, x], 1)
        prob_logit = self.classify_layer(embedding)
        prob_logit_reason = torch.sigmoid(self.classify_layer_reason(embedding))
        prob = self.output_activation(prob_logit)
        return prob_logit, prob_logit_reason, prob