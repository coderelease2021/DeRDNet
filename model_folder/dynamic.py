import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from transformers import DistilBertTokenizer, DistilBertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from math import floor
import torch.nn as nn

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

class CNNFilterBN(nn.Module):
    def __init__(self, config, embeddings):
        super(CNNFilterBN, self).__init__()
        self.convs = nn.ModuleList()
        self.embeddings = embeddings
        self.pad_replace = config.pad_replace
        self.common_conv = nn.Sequential(
            nn.BatchNorm1d(embeddings.word_embeddings.embedding_dim),
            nn.Conv1d(embeddings.word_embeddings.embedding_dim, 1, kernel_size=3,  padding=int(floor(3 / 2)))
        )
        with torch.no_grad():
            self.common_conv[1].bias += 3
        for ith in range(10):
            common_model = nn.ModuleList()
            tmp = nn.BatchNorm1d(embeddings.word_embeddings.embedding_dim)
            common_model.append(tmp)
            for filter_size in config.filter_sizes:
                tmp = nn.Conv1d(embeddings.word_embeddings.embedding_dim, 1, kernel_size=filter_size,  padding=int(floor(filter_size / 2)))
                with torch.no_grad():
                    tmp.bias += (3/len(config.filter_sizes))
                common_model.append(tmp)
            self.convs.append(common_model)
        self.mask_value = 0
        self.pad_embedding = self.embeddings(torch.LongTensor(config.tokenizer(['[UNK]'])['input_ids']))[:, 1, :]
        self.pad_embedding = self.pad_embedding.unsqueeze(0).unsqueeze(-1).to(config.device)

    def module_forward(self, x, model_list):
        x = model_list[0](x)
        results = 0
        for layer in model_list[1:]:
            tmp = layer(x)
            results += tmp
        return results

    def forward(self, x, B):
        short_text = x[0]
        long_text = x[2]

        embeddings_short = self.embeddings(short_text)
        embeddings_short = embeddings_short.view(B, 6, short_text.shape[1], -1)
        embeddings_short = embeddings_short.transpose(2, 3)
        short_mask_sec = []
        for k in range(6):
            tmp = self.module_forward(embeddings_short[:, k], self.convs[k])
            short_mask_sec.append(tmp)
        common_mask = self.common_conv(embeddings_short.view(-1, embeddings_short.shape[2], embeddings_short.shape[3])).view(B, 6, 1, -1)
        short_mask_sec = torch.sigmoid(torch.stack(short_mask_sec, dim=1))*torch.sigmoid(common_mask)
        if self.pad_replace:
            embeddings_short = (embeddings_short * short_mask_sec + (1 - short_mask_sec) * self.pad_embedding).transpose(2, 3)
        else:
            embeddings_short = (embeddings_short*short_mask_sec).transpose(2, 3)
        embeddings_short = embeddings_short.view(-1, embeddings_short.shape[2], embeddings_short.shape[3])

        embeddings_long = self.embeddings(long_text)
        embeddings_long = embeddings_long.view(B, 4, long_text.shape[1], -1)
        embeddings_long = embeddings_long.transpose(2, 3)
        long_mask_sec = []
        for k in range(4):
            tmp = self.module_forward(embeddings_long[:, k], self.convs[k+6])
            long_mask_sec.append(tmp)
        common_mask = self.common_conv(embeddings_long.view(-1, embeddings_long.shape[2], embeddings_long.shape[3])).view(B, 4, 1, -1)
        long_mask_sec = torch.sigmoid(torch.stack(long_mask_sec, dim=1))*torch.sigmoid(common_mask)
        if self.pad_replace:
            embeddings_long = (embeddings_long * long_mask_sec + (1 - long_mask_sec) * self.pad_embedding).transpose(2, 3)
        else:
            embeddings_long = (embeddings_long * long_mask_sec).transpose(2, 3)
        embeddings_long = embeddings_long.view(-1, embeddings_long.shape[2], embeddings_long.shape[3])

        mean_mask = ((short_mask_sec.mean()+long_mask_sec.mean())/2)
        self.mask_value = mean_mask.item()
        return (embeddings_short, embeddings_long), mean_mask


class CNNFilterBNSTD(nn.Module):
    def __init__(self, config, embeddings):
        super(CNNFilterBNSTD, self).__init__()
        self.convs = nn.ModuleList()
        self.embeddings = embeddings
        self.pad_replace = config.pad_replace
        self.common_conv = nn.Sequential(
            nn.BatchNorm1d(embeddings.word_embeddings.embedding_dim),
            nn.Conv1d(embeddings.word_embeddings.embedding_dim, 1, kernel_size=3,  padding=int(floor(3 / 2)))
        )
        for ith in range(10):
            common_model = nn.ModuleList()
            tmp = nn.BatchNorm1d(embeddings.word_embeddings.embedding_dim)
            common_model.append(tmp)
            for filter_size in config.filter_sizes:
                tmp = nn.Conv1d(embeddings.word_embeddings.embedding_dim, 1, kernel_size=filter_size,  padding=int(floor(filter_size / 2)))
                common_model.append(tmp)
            self.convs.append(common_model)
        self.mask_value = 0
        self.pad_embedding = self.embeddings(torch.LongTensor(config.tokenizer(['[UNK]'])['input_ids']))[:, 1, :]
        self.pad_embedding = self.pad_embedding.unsqueeze(0).unsqueeze(-1).to(config.device)

    def module_forward(self, x, model_list):
        x = model_list[0](x)
        results = 0
        for layer in model_list[1:]:
            tmp = layer(x)
            results += tmp
        return results

    def forward(self, x, B):
        short_text = x[0]
        long_text = x[2]

        embeddings_short = self.embeddings(short_text)
        embeddings_short = embeddings_short.view(B, 6, short_text.shape[1], -1)
        embeddings_short = embeddings_short.transpose(2, 3)
        short_mask_sec = []
        for k in range(6):
            tmp = self.module_forward(embeddings_short[:, k], self.convs[k])
            short_mask_sec.append(tmp)
        common_mask = self.common_conv(embeddings_short.view(-1, embeddings_short.shape[2], embeddings_short.shape[3])).view(B, 6, 1, -1)
        short_mask_sec = torch.sigmoid(torch.stack(short_mask_sec, dim=1))*torch.sigmoid(common_mask)
        if self.pad_replace:
            embeddings_short = (embeddings_short * short_mask_sec + (1 - short_mask_sec) * self.pad_embedding).transpose(2, 3)
        else:
            embeddings_short = (embeddings_short*short_mask_sec).transpose(2, 3)
        embeddings_short = embeddings_short.view(-1, embeddings_short.shape[2], embeddings_short.shape[3])

        embeddings_long = self.embeddings(long_text)
        embeddings_long = embeddings_long.view(B, 4, long_text.shape[1], -1)
        embeddings_long = embeddings_long.transpose(2, 3)
        long_mask_sec = []
        for k in range(4):
            tmp = self.module_forward(embeddings_long[:, k], self.convs[k+6])
            long_mask_sec.append(tmp)
        common_mask = self.common_conv(embeddings_long.view(-1, embeddings_long.shape[2], embeddings_long.shape[3])).view(B, 4, 1, -1)
        long_mask_sec = torch.sigmoid(torch.stack(long_mask_sec, dim=1))*torch.sigmoid(common_mask)
        if self.pad_replace:
            embeddings_long = (embeddings_long * long_mask_sec + (1 - long_mask_sec) * self.pad_embedding).transpose(2, 3)
        else:
            embeddings_long = (embeddings_long * long_mask_sec).transpose(2, 3)
        embeddings_long = embeddings_long.view(-1, embeddings_long.shape[2], embeddings_long.shape[3])

        mean_mask = ((short_mask_sec.mean()+long_mask_sec.mean())/2)
        self.mask_value = mean_mask.item()
        return (embeddings_short, embeddings_long), mean_mask

class TextEncoderFilterBN(nn.Module):
    def __init__(self, config):
        super(TextEncoderFilterBN, self).__init__()
        self.bert = DistilBertModel.from_pretrained(config.bert_path)
        self.embeddings = self.bert.embeddings
        self.cnn_filter = CNNFilterBN(config, self.embeddings)

    def forward(self, x, B):
        short_mask = x[1]
        long_mask = x[3]
        (embeddings_short, embeddings_long), mask_mean = self.cnn_filter(x, B)
        short_outputs = self.bert(inputs_embeds=embeddings_short, attention_mask=short_mask)
        long_outputs = self.bert(inputs_embeds=embeddings_long, attention_mask=long_mask)
        return short_outputs[0][:, 0, :], long_outputs[0][:, 0, :], mask_mean

class TextEncoderFilterBNSTD(nn.Module):
    def __init__(self, config):
        super(TextEncoderFilterBNSTD, self).__init__()
        self.bert = DistilBertModel.from_pretrained(config.bert_path)
        self.embeddings = self.bert.embeddings
        self.cnn_filter = CNNFilterBNSTD(config, self.embeddings)

    def forward(self, x, B):
        short_mask = x[1]
        long_mask = x[3]
        (embeddings_short, embeddings_long), mask_mean = self.cnn_filter(x, B)
        short_outputs = self.bert(inputs_embeds=embeddings_short, attention_mask=short_mask)
        long_outputs = self.bert(inputs_embeds=embeddings_long, attention_mask=long_mask)
        return short_outputs[0][:, 0, :], long_outputs[0][:, 0, :], mask_mean

class EncoderFilterBN(nn.Module):
    def __init__(self, config):
        super(EncoderFilterBN, self).__init__()
        self.text_encoder = TextEncoderFilterBN(config)
        self.data_encoder = DataEncoder(config)
        self.trans_layer = nn.Linear(768, config.emb_dim)
        self.drop_layer = nn.Dropout(0.5)
        self.trans_activation = nn.Tanh()
    def forward(self, data, text):
        data_embedding = self.drop_layer(self.data_encoder(data))
        text_embedding_short, text_embedding_long, mask_mean = self.text_encoder(text, data.shape[0])
        B = data_embedding.shape[0]
        D = text_embedding_short.shape[1]
        text_embedding_short = text_embedding_short.view(B, -1, D)
        #text_embedding_short = torch.sum(text_embedding_short, dim=1)
        text_embedding_long = text_embedding_long.view(B, -1, D)
        #text_embedding_long = torch.sum(text_embedding_long, dim=1)
        text_embedding = torch.cat([text_embedding_short, text_embedding_long], dim=1)
        text_embedding = self.trans_layer(text_embedding)
        text_embedding = self.drop_layer(self.trans_activation(text_embedding))
        #total_embedding = torch.cat([data_embedding, text_embedding], 1)
        return data_embedding, text_embedding, mask_mean

class EncoderFilterBNSTD(nn.Module):
    def __init__(self, config):
        super(EncoderFilterBNSTD, self).__init__()
        self.text_encoder = TextEncoderFilterBNSTD(config)
        self.data_encoder = DataEncoder(config)
        self.trans_layer = nn.Linear(768, config.emb_dim)
        self.drop_layer = nn.Dropout(0.5)
        self.trans_activation = nn.Tanh()
    def forward(self, data, text):
        data_embedding = self.drop_layer(self.data_encoder(data))
        text_embedding_short, text_embedding_long, mask_mean = self.text_encoder(text, data.shape[0])
        B = data_embedding.shape[0]
        D = text_embedding_short.shape[1]
        text_embedding_short = text_embedding_short.view(B, -1, D)
        #text_embedding_short = torch.sum(text_embedding_short, dim=1)
        text_embedding_long = text_embedding_long.view(B, -1, D)
        #text_embedding_long = torch.sum(text_embedding_long, dim=1)
        text_embedding = torch.cat([text_embedding_short, text_embedding_long], dim=1)
        text_embedding = self.trans_layer(text_embedding)
        text_embedding = self.drop_layer(self.trans_activation(text_embedding))
        #total_embedding = torch.cat([data_embedding, text_embedding], 1)
        return data_embedding, text_embedding, mask_mean

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)

class FeatureReaderInitial(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_tasks,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
        d_bn=True,
    ):
        super(FeatureReaderInitial, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim*n_tasks)
        self.n_tasks = n_tasks
        self.d_bn = d_bn
        if self.d_bn:
            self.bn = GBN(
                output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.n_tasks, -1)
        if self.d_bn:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        x = self.act(x)
        return x


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return
class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        d_bn=True,
    ):
        """
        Initialize an attention transformer.
        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(
            output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        elif mask_type == "softmax":
            # Entmax
            self.selector = nn.Softmax(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )
        self.d_bn = d_bn
        if self.d_bn:
            self.bn = GBN(
                output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )

    def forward(self, processed_feat):
        x = self.fc(processed_feat)
        x = self.selector(x)
        return x

from torch.nn.modules.rnn import GRUCell
class DecisionCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DecisionCell, self).__init__()
        self.w_z = nn.Linear(input_size+hidden_size, hidden_size)
        self.w_update = nn.Linear(input_size+hidden_size, hidden_size)

    def forward(self, input, hx):
        x = torch.cat([input, hx], -1)
        z = torch.sigmoid(self.w_z(x))
        update = torch.tanh(self.w_update(x))
        return (1-z)*hx + z*update


class DynamicDecision(nn.Module):
    def __init__(self, config):
        super(DynamicDecision, self).__init__()
        self.encoder = Encoder(config)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)
        self.n_d = config.d_emb_dim
        self.final = nn.Linear(self.n_d, 6)
        xavier_uniform(self.final.weight)
        self.momentum = 0.02
        self.n_steps = config.d_step
        self.n_independent = 1
        self.virtual_batch_size = 8
        self.n_tasks = 6
        self.n_sections = 11
        self.input_dim = config.emb_dim
        self.d_method = config.d_method
        self.bn_initial = GBN(config.emb_dim, virtual_batch_size=self.virtual_batch_size)

        self.initial_splitter = FeatureReaderInitial(
            self.input_dim,
            self.n_d,
            self.n_tasks,
            None,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            d_bn=config.d_bn,
        )

        self.att_transformer = AttentiveTransformer(
                self.n_d,
                self.n_sections,
                virtual_batch_size=self.virtual_batch_size,
                mask_type='softmax',
                momentum=self.momentum,
                d_bn=config.d_bn,
            )
        self.feat_rnn = DecisionCell(config.emb_dim, self.n_d)


    def forward(self, data, text):
        data_embedding, text_embedding = self.encoder(data, text)
        x = torch.cat([data_embedding.unsqueeze(1), text_embedding], 1)
        x = x.transpose(1, 2)
        # initial decision
        x = self.bn_initial(x)
        x = x.transpose(1, 2)
        decision_emb = self.initial_splitter(x.mean(1))
        y = 0
        y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
        y += y_d
        for step in range(self.n_steps):
            M = self.att_transformer(decision_emb)
            # update prior
            # output
            masked_x = M.matmul(x)
            masked_x = masked_x.view(-1, masked_x.shape[2])
            decision_emb = decision_emb.reshape(-1, decision_emb.shape[2])
            decision_emb = self.feat_rnn(masked_x, decision_emb).view(data_embedding.shape[0], -1, decision_emb.shape[1])
            y_d = self.final.weight.mul(decision_emb).sum(dim=2)
            y += y_d
            # update attention
        if self.d_method == 'avg':
            y /= self.n_steps
        elif self.d_method == 'final':
            y = y_d
        prob_logit = y[:, 4:]
        prob_reason = torch.sigmoid(y[:, 0:4])
        prob = self.output_activation(prob_logit)
        binary_loss = 0 * - torch.mean(prob_reason * prob_reason)

        return prob_logit, prob_reason, prob, None, binary_loss

class DynamicDecisionFilter(nn.Module):
    def __init__(self, config):
        super(DynamicDecisionFilter, self).__init__()
        self.encoder = EncoderFilterBN(config)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)
        self.n_d = config.d_emb_dim
        self.final = nn.Linear(self.n_d, 6)
        xavier_uniform(self.final.weight)
        self.momentum = 0.02
        self.n_steps = config.d_step
        self.n_independent = 1
        self.virtual_batch_size = 8
        self.n_tasks = 6
        self.n_sections = 11
        self.input_dim = config.emb_dim
        self.d_method = config.d_method
        self.bn_initial = GBN(config.emb_dim, virtual_batch_size=self.virtual_batch_size)

        self.initial_splitter = FeatureReaderInitial(
            self.input_dim,
            self.n_d,
            self.n_tasks,
            None,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            d_bn=config.d_bn,
        )

        self.att_transformer = AttentiveTransformer(
                self.n_d,
                self.n_sections,
                virtual_batch_size=self.virtual_batch_size,
                mask_type='softmax',
                momentum=self.momentum,
                d_bn=config.d_bn,
            )
        self.feat_rnn = DecisionCell(config.emb_dim, self.n_d)


    def forward(self, data, text):
        data_embedding, text_embedding, mask_mean = self.encoder(data, text)
        x = torch.cat([data_embedding.unsqueeze(1), text_embedding], 1)
        x = x.transpose(1, 2)
        # initial decision
        x = self.bn_initial(x)
        x = x.transpose(1, 2)
        decision_emb = self.initial_splitter(x.mean(1))
        y = 0
        y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
        y += y_d
        for step in range(self.n_steps):
            M = self.att_transformer(decision_emb)
            # update prior
            # output
            masked_x = M.matmul(x)
            masked_x = masked_x.view(-1, masked_x.shape[2])
            decision_emb = decision_emb.reshape(-1, decision_emb.shape[2])
            decision_emb = self.feat_rnn(masked_x, decision_emb).view(data_embedding.shape[0], -1, decision_emb.shape[1])
            y_d = self.final.weight.mul(decision_emb).sum(dim=2)
            y += y_d
            # update attention
        if self.d_method == 'avg':
            y /= self.n_steps
        elif self.d_method == 'final':
            y = y_d
        prob_logit = y[:, 4:]
        prob_reason = torch.sigmoid(y[:, 0:4])
        prob = self.output_activation(prob_logit)
        binary_loss = 0 * - torch.mean(prob_reason * prob_reason)

        return prob_logit, prob_reason, prob, None, mask_mean

class DynamicDecisionFilterGate(nn.Module):
    def __init__(self, config, eva=False):
        super(DynamicDecisionFilterGate, self).__init__()
        self.evaluation = eva
        self.encoder = EncoderFilterBN(config)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)
        self.n_d = config.d_emb_dim
        self.final = nn.Linear(self.n_d, 6)
        self.final_gate = nn.Linear(self.n_d, 6)
        xavier_uniform(self.final.weight)
        self.momentum = 0.02
        self.n_steps = config.d_step
        self.n_independent = 1
        self.virtual_batch_size = 8
        self.n_tasks = 6
        self.n_sections = 11
        self.input_dim = config.emb_dim
        self.d_method = config.d_method
        self.bn_initial = GBN(config.emb_dim, virtual_batch_size=self.virtual_batch_size)

        self.initial_splitter = FeatureReaderInitial(
            self.input_dim,
            self.n_d,
            self.n_tasks,
            None,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            d_bn=config.d_bn,
        )

        self.att_transformer = AttentiveTransformer(
                self.n_d,
                self.n_sections,
                virtual_batch_size=self.virtual_batch_size,
                mask_type='softmax',
                momentum=self.momentum,
                d_bn=config.d_bn,
            )
        self.feat_rnn = DecisionCell(config.emb_dim, self.n_d)


    def forward(self, data, text):
        data_embedding, text_embedding, mask_mean = self.encoder(data, text)
        x = torch.cat([data_embedding.unsqueeze(1), text_embedding], 1)
        x = x.transpose(1, 2)
        # initial decision
        x = self.bn_initial(x)
        x = x.transpose(1, 2)
        decision_emb = self.initial_splitter(x.mean(1))
        y = []
        g = []
        ms = []
        y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
        g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
        y.append(y_d)
        g.append(g_d)
        for step in range(self.n_steps):
            M = self.att_transformer(decision_emb)
            ms.append(M.detach().cpu())
            # update prior
            # output
            masked_x = M.matmul(x)
            masked_x = masked_x.view(-1, masked_x.shape[2])
            decision_emb = decision_emb.reshape(-1, decision_emb.shape[2])
            decision_emb = self.feat_rnn(masked_x, decision_emb).view(data_embedding.shape[0], -1, decision_emb.shape[1])
            y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
            g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
            y.append(y_d)
            g.append(g_d)
            # update attention
        if self.d_method == 'avg':
            y = torch.stack(y, 2)
            g = torch.stack(g, 2)
            g = torch.softmax(g, dim=-1)
            y = (y * g).sum(dim=-1)

        elif self.d_method == 'final':
            y = y_d
        prob_logit = y[:, 4:]
        prob_reason = torch.sigmoid(y[:, 0:4])
        prob = self.output_activation(prob_logit)
        if self.evaluation:
            return prob_logit, prob_reason, prob, (ms, g), mask_mean
        return prob_logit, prob_reason, prob, None, mask_mean

class DynamicDecisionFilterGateSTD(nn.Module):
    def __init__(self, config, eva=False):
        super(DynamicDecisionFilterGateSTD, self).__init__()
        self.evaluation = eva
        self.encoder = EncoderFilterBNSTD(config)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)
        self.n_d = config.d_emb_dim
        self.final = nn.Linear(self.n_d, 6)
        self.final_gate = nn.Linear(self.n_d, 6)
        xavier_uniform(self.final.weight)
        self.momentum = 0.02
        self.n_steps = config.d_step
        self.n_independent = 1
        self.virtual_batch_size = 8
        self.n_tasks = 6
        self.n_sections = 11
        self.input_dim = config.emb_dim
        self.d_method = config.d_method
        self.bn_initial = GBN(config.emb_dim, virtual_batch_size=self.virtual_batch_size)

        self.initial_splitter = FeatureReaderInitial(
            self.input_dim,
            self.n_d,
            self.n_tasks,
            None,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            d_bn=config.d_bn,
        )

        self.att_transformer = AttentiveTransformer(
                self.n_d,
                self.n_sections,
                virtual_batch_size=self.virtual_batch_size,
                mask_type='softmax',
                momentum=self.momentum,
                d_bn=config.d_bn,
            )
        self.feat_rnn = DecisionCell(config.emb_dim, self.n_d)


    def forward(self, data, text):
        data_embedding, text_embedding, mask_mean = self.encoder(data, text)
        x = torch.cat([data_embedding.unsqueeze(1), text_embedding], 1)
        x = x.transpose(1, 2)
        # initial decision
        x = self.bn_initial(x)
        x = x.transpose(1, 2)
        decision_emb = self.initial_splitter(x.mean(1))
        y = []
        g = []
        ms = []
        y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
        g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
        y.append(y_d)
        g.append(g_d)
        for step in range(self.n_steps):
            M = self.att_transformer(decision_emb)
            ms.append(M.detach().cpu())
            # update prior
            # output
            masked_x = M.matmul(x)
            masked_x = masked_x.view(-1, masked_x.shape[2])
            decision_emb = decision_emb.reshape(-1, decision_emb.shape[2])
            decision_emb = self.feat_rnn(masked_x, decision_emb).view(data_embedding.shape[0], -1, decision_emb.shape[1])
            y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
            g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
            y.append(y_d)
            g.append(g_d)
            # update attention
        if self.d_method == 'avg':
            y = torch.stack(y, 2)
            g = torch.stack(g, 2)
            g = torch.softmax(g, dim=-1)
            y = (y * g).sum(dim=-1)

        elif self.d_method == 'final':
            y = y_d
        prob_logit = y[:, 4:]
        prob_reason = torch.sigmoid(y[:, 0:4])
        prob = self.output_activation(prob_logit)
        if self.evaluation:
            return prob_logit, prob_reason, prob, (ms, g), mask_mean
        return prob_logit, prob_reason, prob, None, mask_mean


class DynamicDecisionGate(nn.Module):
    def __init__(self, config, eva=False):
        super(DynamicDecisionGate, self).__init__()
        self.evaluation = eva
        self.encoder = Encoder(config)
        self.output_activation = nn.Softmax(dim=-1)
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_reason = nn.BCELoss(reduce=False)
        self.n_d = config.d_emb_dim
        self.final = nn.Linear(self.n_d, 6)
        self.final_gate = nn.Linear(self.n_d, 6)
        xavier_uniform(self.final.weight)
        self.momentum = 0.02
        self.n_steps = config.d_step
        self.n_independent = 1
        self.virtual_batch_size = 8
        self.n_tasks = 6
        self.n_sections = 11
        self.input_dim = config.emb_dim
        self.d_method = config.d_method
        self.bn_initial = GBN(config.emb_dim, virtual_batch_size=self.virtual_batch_size)

        self.initial_splitter = FeatureReaderInitial(
            self.input_dim,
            self.n_d,
            self.n_tasks,
            None,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            d_bn=config.d_bn,
        )

        self.att_transformer = AttentiveTransformer(
                self.n_d,
                self.n_sections,
                virtual_batch_size=self.virtual_batch_size,
                mask_type='softmax',
                momentum=self.momentum,
                d_bn=config.d_bn,
            )
        self.feat_rnn = DecisionCell(config.emb_dim, self.n_d)


    def forward(self, data, text):
        data_embedding, text_embedding = self.encoder(data, text)
        x = torch.cat([data_embedding.unsqueeze(1), text_embedding], 1)
        x = x.transpose(1, 2)
        # initial decision
        x = self.bn_initial(x)
        x = x.transpose(1, 2)
        decision_emb = self.initial_splitter(x.mean(1))
        y = []
        g = []
        ms = []
        y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
        g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
        y.append(y_d)
        g.append(g_d)
        for step in range(self.n_steps):
            M = self.att_transformer(decision_emb)
            ms.append(M.detach().cpu())
            # update prior
            # output
            masked_x = M.matmul(x)
            masked_x = masked_x.view(-1, masked_x.shape[2])
            decision_emb = decision_emb.reshape(-1, decision_emb.shape[2])
            decision_emb = self.feat_rnn(masked_x, decision_emb).view(data_embedding.shape[0], -1, decision_emb.shape[1])
            y_d = self.final.weight.mul(decision_emb).sum(dim=2).add(self.final.bias)
            g_d = self.final_gate.weight.mul(decision_emb).sum(dim=2).add(self.final_gate.bias)
            y.append(y_d)
            g.append(g_d)
            # update attention
        if self.d_method == 'avg':
            y = torch.stack(y, 2)
            g = torch.stack(g, 2)
            g = torch.softmax(g, dim=-1)
            y = (y * g).sum(dim=-1)

        elif self.d_method == 'final':
            y = y_d
        prob_logit = y[:, 4:]
        prob_reason = torch.sigmoid(y[:, 0:4])
        prob = self.output_activation(prob_logit)
        mask_mean = 0 * - torch.mean(prob_reason * prob_reason)
        if self.evaluation:
            return prob_logit, prob_reason, prob, (ms, g), mask_mean
        return prob_logit, prob_reason, prob, None, mask_mean