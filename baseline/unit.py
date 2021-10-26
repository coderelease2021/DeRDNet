import torch
from transformers import XLNetTokenizer
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from difflib import SequenceMatcher
from torch.utils.data import Dataset, DataLoader
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
'''
import torch
from transformers import XLNetTokenizer, BertTokenizer
from baseline.unit import collate_fn_han
from baseline.Han import BaseModel
class Config(object):

    """配置参数"""
    def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.bert_path = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = BaseModel
        self.collate_fn = collate_fn_han
        self.data_dim = 128
        self.emb_dim = 256
        self.batch_size = batch_size
        self.warmup_lr = 1e-5
        self.data = 'small_data_cpl.pkl'
        self.lr = 1e-4
        self.max_text_cut = 1024
        self.warmup_epoch = 1
        self.train_epoch = 10
config = Config(16)
'''
def collate_fn_han(train_data):
    value_filed = [data[0][0] for data in train_data]
    value_filed = torch.tensor(value_filed, dtype=torch.long, device=config.device)
    text_field = [data[0][1] for data in train_data]
    text_field_sen = []
    text_field_sen_len = []
    for para in text_field:
        sentences = sent_tokenize(para)
        while len(sentences) < 25:
            sentences.append('')
        if len(sentences) > 25:
            sentences = sentences[0:25]
        text_field_sen += sentences
        text_field_sen_len.append(len(sentences))
    encoding = config.tokenizer(text_field_sen, return_tensors='pt', padding=True, truncation=True, max_length=70)
    input_ids = encoding['input_ids'].to(config.device)
    input_ids = input_ids.view(len(text_field), 25, -1)
    attention_mask = encoding['attention_mask'].to(config.device)
    label = [data[1][0] for data in train_data]
    label_reason_mask = [data[1][2] for data in train_data]
    label_reason = [data[1][3] for data in train_data]
    label = torch.cuda.LongTensor(label)
    label_reason_mask = torch.cuda.FloatTensor(label_reason_mask)
    label_reason = torch.cuda.FloatTensor(label_reason)
    return value_filed, input_ids, attention_mask, label, label_reason, label_reason_mask

def collate_fn(train_data):
    value_filed = [data[0][0] for data in train_data]
    value_filed = torch.tensor(value_filed, dtype=torch.long, device=config.device)
    text_field = [data[0][1] for data in train_data]
    encoding = config.tokenizer(text_field, return_tensors='pt', padding=True, truncation=True, max_length=config.max_text_cut)
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    label = [data[1][0] for data in train_data]
    label_reason_mask = [data[1][2] for data in train_data]
    label_reason = [data[1][3] for data in train_data]
    label = torch.cuda.LongTensor(label)
    label_reason_mask = torch.cuda.FloatTensor(label_reason_mask)
    label_reason = torch.cuda.FloatTensor(label_reason)
    return value_filed, input_ids, attention_mask, label, label_reason, label_reason_mask

from baseline.environment import config
key_map = {0:1, 1:3, 2:3, 3:3, 4:2, 5:2, 6:1, 7:1, 8:0, 9:2, 10:4}
class MyData(Dataset):
    def __init__(self, data_full, config):
        self.source = []
        self.target = []
        self.config = config

        self.dmap = {'sex': {'All': 1, 'Male': 2, 'Female': 3, 'None': 4},
                     'study_type': {'All': 5, 'Interventional': 6, 'Observational': 7, 'Expanded Access': 8, 'None': 9},
                     'healthy_volunteers': {'Yes': 10, 'No': 11, 'Accepts Healthy Volunteers': 12, 'None': 13},
                     'sponsor_type': {'NIH': 14, 'Other U.S. Federal agency': 15, 'Industry': 16, 'others': 17, 'None': 18}}
        self.age_group = {'teen': 19, 'adult': 20, 'old': 21, 'None': 22}
        length_count = []
        d_count = 0
        for case in tqdm(data_full):
            data_field = []
            len_field = []
            for key, map_dict in self.dmap.items():
                text_ans = case['Discrete'][key]
                if text_ans in map_dict:
                    data_field.append(map_dict[text_ans])
                else:
                    names = map_dict.keys()
                    max_score = 0

                    for name in names:
                        score = similarity(text_ans, name)
                        if score > max_score:
                            max_score = score
                            ans_tar = name
                    if max_score == 0:
                        ans_tar = 'None'
                    data_field.append(map_dict[ans_tar])
            # age group
            try:
                min_age = float(case['Discrete']['min_age'].split(' ')[0])
                max_age = float(case['Discrete']['max_age'].split(' ')[0])
                if min_age < 18:
                    data_field.append(self.age_group['teen'])
                else:
                    data_field.append(0)
                if min_age < 65 and max_age >= 18:
                    data_field.append(self.age_group['adult'])
                else:
                    data_field.append(0)
                if max_age >= 65:
                    data_field.append(self.age_group['old'])
                else:
                    data_field.append(0)
            except:
                data_field.append(self.age_group['None'])
                data_field.append(self.age_group['None'])
                data_field.append(self.age_group['None'])
            text = ''
            for x in list(case['Text'].values()):
                text += x
            self.source.append([data_field, text])
            if case['state'] in ['Completed', 'Available', 'Approved for marketing']:
                state = 0
            else:
                state = 1
            if case['why_stopped'] == 'None':
                state_reason_mask = 0
            else:
                state_reason_mask = 1
            state_reason = case['why_stopped']
            if 'simplified_reason_new' in case:
                simplified_reason = case['simplified_reason_new']
                simplified_reason_mask = 1
            else:
                simplified_reason = [0 for x in range(4)]
                simplified_reason_mask = 0
            self.target.append([state, state_reason, state_reason_mask, simplified_reason, simplified_reason_mask])
        print("D count:%d" % d_count)
    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item], self.target[item]


