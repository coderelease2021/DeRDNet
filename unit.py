import torch
from transformers import BertTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pickle
from tqdm import tqdm
from difflib import SequenceMatcher
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset, DataLoader
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
'''
import torch
from transformers import DistilBertTokenizer
class Config(object):

    """配置参数"""
        def __init__(self, batch_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.bert_path = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.model = BaseModelAttentionHidden
        self.reason_tokenizer = None
        self.reason_vocab_size = None
        self.data_dim = 128
        self.emb_dim = 256
        self.hidden_dim = 64
        self.reason_length = 25
        self.batch_size = batch_size
        self.warmup_lr = 1e-5
        self.data = 'small_data_cpl.pkl'
        self.lr = 1e-4
        self.max_text_cut = 300
        self.warmup_epoch = 1
        self.train_epoch = 10
config = Config(16)
'''
from environment import config
def train_tokenizer(reasons):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(reasons, trainer)
    return tokenizer
key_map = {0:1, 1:3, 2:3, 3:3, 4:2, 5:2, 6:1, 7:1, 8:0, 9:2, 10:4}
class MyData(Dataset):
    def __init__(self, data_full, config):
        if config.use_simple_reason:
            ans_length = 4
        else:
            ans_length = 11
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
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
            if 'simplified_reason_new' in case and case['simplified_reason_new'][10] == 1:
                d_count += 1
                continue
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

            group_100 = [list(case['Text'].values())[x] for x in [0, 2, 3, 4, 7, 9]]
            group_600 = [list(case['Text'].values())[x] for x in [1, 5, 6, 8]]
            self.source.append([data_field, [group_100, group_600]])
            if case['state'] in ['Completed', 'Available', 'Approved for marketing']:
                state = 0
            else:
                state = 1
            if case['why_stopped'] == 'None':
                state_reason_mask = 0
                state_reason = ''
            else:
                state_reason_mask = 1
                state_reason = case['why_stopped']
            if 'stop_reason' in case:
                simplified_reason = case['stop_reason']
                simplified_reason_mask = 1
            else:
                simplified_reason = [0 for x in range(ans_length)]
                simplified_reason_mask = 0
            self.target.append([state, state_reason, state_reason_mask, simplified_reason, simplified_reason_mask])
        print("D count:%d" %d_count)
    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item], self.target[item]


def collate_fn(train_data):
    value_filed = [data[0][0] for data in train_data]
    value_filed = torch.tensor(value_filed, dtype=torch.long, device=config.device)
    text_field = [data[0][1] for data in train_data]
    text_field_short = [field[0] for field in text_field]
    text_field_long = [field[1] for field in text_field]
    text_field_short = np.concatenate(text_field_short, axis=0)
    text_field_long = np.concatenate(text_field_long, axis=0)
    text_reason = [data[1][1] for data in train_data]

    encoding = config.tokenizer(text_field_short.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=100)
    input_ids_short = encoding['input_ids'].to(config.device)
    attention_mask_short = encoding['attention_mask'].to(config.device)
    encoding = config.tokenizer(text_field_long.tolist(), return_tensors='pt', padding=True, truncation=True,
                                max_length=config.max_text_cut)
    input_ids_long = encoding['input_ids'].to(config.device)
    attention_mask_long = encoding['attention_mask'].to(config.device)

    config.reason_tokenizer.enable_padding(length=config.reason_length)
    config.reason_tokenizer.enable_truncation(max_length=config.reason_length)
    encoding = config.reason_tokenizer.encode_batch(text_reason)
    reason_ids = [x.ids for x in encoding]
    attention_mask_reason = [x.attention_mask for x in encoding]
    reason_ids = torch.cuda.LongTensor(reason_ids)
    attention_mask_reason = torch.cuda.FloatTensor(attention_mask_reason)


    label = [data[1][0] for data in train_data]
    label_reason_mask = [data[1][4] for data in train_data]
    label_reason = [data[1][3] for data in train_data]
    label = torch.LongTensor(label).to(config.device)
    label_reason_mask = torch.FloatTensor(label_reason_mask).to(config.device)
    label_reason = torch.FloatTensor(label_reason).to(config.device)
    return value_filed, input_ids_short, attention_mask_short, input_ids_long, attention_mask_long, label, label_reason, label_reason_mask, reason_ids, attention_mask_reason

def collate_fn_eval(train_data):
    value_filed = [data[0][0] for data in train_data]
    value_filed = torch.tensor(value_filed, dtype=torch.long, device=config.device)
    text_field = [data[0][1] for data in train_data]
    text_field_short = [field[0] for field in text_field]
    text_field_long = [field[1] for field in text_field]
    text_field_short = np.concatenate(text_field_short, axis=0)
    text_field_long = np.concatenate(text_field_long, axis=0)
    text_reason = [data[1][1] for data in train_data]

    encoding = config.tokenizer(text_field_short.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=100)
    input_ids_short = encoding['input_ids'].to(config.device)
    attention_mask_short = encoding['attention_mask'].to(config.device)
    encoding = config.tokenizer(text_field_long.tolist(), return_tensors='pt', padding=True, truncation=True,
                                max_length=config.max_text_cut)
    input_ids_long = encoding['input_ids'].to(config.device)
    attention_mask_long = encoding['attention_mask'].to(config.device)

    config.reason_tokenizer.enable_padding(length=config.reason_length)
    config.reason_tokenizer.enable_truncation(max_length=config.reason_length)
    encoding = config.reason_tokenizer.encode_batch(text_reason)
    reason_ids = [x.ids for x in encoding]
    attention_mask_reason = [x.attention_mask for x in encoding]
    reason_ids = torch.cuda.LongTensor(reason_ids)
    attention_mask_reason = torch.cuda.FloatTensor(attention_mask_reason)


    label = [data[1][0] for data in train_data]
    label_reason_mask = [data[1][4] for data in train_data]
    label_reason = [data[1][3] for data in train_data]
    label = torch.LongTensor(label).to(config.device)
    label_reason_mask = torch.FloatTensor(label_reason_mask).to(config.device)
    label_reason = torch.FloatTensor(label_reason).to(config.device)
    return value_filed, input_ids_short, attention_mask_short, input_ids_long, attention_mask_long, label, label_reason, label_reason_mask, reason_ids, attention_mask_reason, (text_field_short, text_field_long)

def collate_fn_alpha(train_data):
    value_filed = [data[0][0] for data in train_data]
    value_filed = torch.tensor(value_filed, dtype=torch.long, device=config.device)
    text_field = [data[0][1] for data in train_data]
    text_field_short = [field[0] for field in text_field]
    text_field_long = [field[1] for field in text_field]
    text_field_short = np.concatenate(text_field_short, axis=0)
    text_field_long = np.concatenate(text_field_long, axis=0)
    text_reason = [data[1][1] for data in train_data]
    text_title = [data[1][5] for data in train_data]

    encoding = config.tokenizer(text_field_short.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=100)
    input_ids_short = encoding['input_ids'].to(config.device)
    attention_mask_short = encoding['attention_mask'].to(config.device)
    encoding = config.tokenizer(text_field_long.tolist(), return_tensors='pt', padding=True, truncation=True,
                                max_length=config.max_text_cut)
    input_ids_long = encoding['input_ids'].to(config.device)
    attention_mask_long = encoding['attention_mask'].to(config.device)

    config.reason_tokenizer.enable_padding(length=config.reason_length)
    config.reason_tokenizer.enable_truncation(max_length=config.reason_length)
    encoding = config.reason_tokenizer.encode_batch(text_reason)
    reason_ids = [x.ids for x in encoding]
    attention_mask_reason = [x.attention_mask for x in encoding]
    reason_ids = torch.cuda.LongTensor(reason_ids)
    attention_mask_reason = torch.cuda.FloatTensor(attention_mask_reason)
    encoding = config.reason_tokenizer.encode_batch(text_title)
    title_ids = [x.ids for x in encoding]
    attention_mask_title = [x.attention_mask for x in encoding]
    title_ids = torch.cuda.LongTensor(title_ids)
    attention_mask_title = torch.cuda.FloatTensor(attention_mask_title)

    label = [data[1][0] for data in train_data]
    label_reason_mask = [data[1][4] for data in train_data]
    label_reason = [data[1][3] for data in train_data]
    label = torch.cuda.LongTensor(label)
    label_reason_mask = torch.cuda.FloatTensor(label_reason_mask)
    label_reason = torch.cuda.FloatTensor(label_reason)
    return value_filed, input_ids_short, attention_mask_short, input_ids_long, attention_mask_long, label, \
            label_reason, label_reason_mask, reason_ids, attention_mask_reason, title_ids, attention_mask_title



