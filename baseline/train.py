import torch
import numpy as np
import pickle
from transformers import AdamW
from baseline.unit import MyData, DataLoader, collate_fn
from baseline.unit import config
from tqdm import tqdm
from baseline.XLNet import Encoder, BaseModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
def build(config):
    raw_data = pickle.load(open(config.data, 'rb'))
    len_train = len(raw_data) - 100
    reasons = []
    for case in raw_data[0:len_train]:
        reasons.append(case['why_stopped'])
    dataset = MyData(raw_data[0:len_train], config)
    dataset_val = MyData(raw_data[len_train:len_train + 50], config)
    dataset_test = MyData(raw_data[len_train + 50:], config)
    train_dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size
                                  , collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset_val, batch_size=config.batch_size
                                , collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=config.batch_size
                                 , collate_fn=collate_fn)
    model = config.model(config).cuda()
    model_bert = model.encoder.text_encoder
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model_bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_bert = AdamW(optimizer_grouped_parameters, lr=config.lr)
    name_list = []
    for name, value in model_bert.named_parameters():
        name_list.append(name)
    params = [{'params': [p for n, p in model.named_parameters() if not 'text_encoder' in n]}]
    optimizer = AdamW(params, lr=config.lr)
    return model, optimizer, optimizer_bert, train_dataloader, val_dataloader, test_dataloader


def train(model, optimizer_bert, optimizer, train_dataloder, val_dataloader, test_dataloader, config, record):

    min_loss = 1000
    t_count = 0
    for epoch in range(config.train_epoch):
        if epoch == 0 and epoch < config.warmup_epoch:
            for param_group in optimizer_bert.param_groups:
                param_group['lr'] = config.warmup_lr
        if epoch == config.warmup_epoch:
            for param_group in optimizer_bert.param_groups:
                param_group['lr'] = config.lr
        for step, (value_filed, input_ids, attention_mask, label, label_reason, label_reason_mask) in tqdm(enumerate(train_dataloder)):
            prob_logit, prob_reason_logit, prob = model(value_filed, (input_ids, attention_mask))
            loss = model.loss_func(prob_logit, label)
            loss_reason = model.loss_func_reason(prob_reason_logit, label_reason)
            loss_reason = torch.mean(loss_reason * label_reason_mask.unsqueeze(dim=1))
            loss = loss + loss_reason
            optimizer.zero_grad()
            optimizer_bert.zero_grad()
            (loss).backward()
            optimizer.step()
            if step % 100 == 0:
                print("Epoch: %d Step: %d Loss: %f Loss Reason: %f" %(epoch, step, (loss).item(), loss_reason.item()))
        _, _, _, test_loss, _ = test(model, optimizer_bert, optimizer, val_dataloader, config, record, False)
        if test_loss < min_loss:
            print('New Test Loss:%f' % test_loss)
            t_count = 1
            min_loss = test_loss
            _, _, _, test_loss, result_one = test(model, optimizer_bert, optimizer, test_dataloader, config, record, True)
            best_result_one = result_one
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            #torch.save(state, 'model.bin')
        else:
            t_count += 1
        if t_count > 3:
            print("Early Stop")
            break
    return best_result_one

import re
def varname(p):
    m = re.findall(r'\'(.*)\'', str(config.model))
    if m:
        return m[0]

def test(model, optimizer_bert, optimizer, test_dataloder, config, record, is_test):
    if is_test:
        print('Testing')
    else:
        print('Validation')
    model.eval()
    raw_prediction = {}
    prediction = []
    answer = []
    prediction_reason = []
    answer_reason = []
    raw_state_prob = []
    raw_risk_prob = []
    avg_loss = []
    answer_score = []
    for step, (value_filed, input_ids, attention_mask, label, label_reason, label_reason_mask) in enumerate(
            test_dataloder):
        prob_logit, prob_reason_logit, prob = model(value_filed, (input_ids, attention_mask))
        loss = model.loss_func(prob_logit, label)
        avg_loss.append(loss.item())
        pre_ans = torch.argmax(prob, dim=-1).detach().cpu().numpy()
        raw_state_prob.append(prob.detach().cpu().numpy())
        pre_ans_reason = prob_reason_logit.detach().cpu().numpy() * np.expand_dims(pre_ans, 1)
        answer.append(label.cpu().numpy())
        answer_score.append(prob[:, 1].detach().cpu().numpy())
        prediction.append(pre_ans)
        if config.neg_test:
            index = label_reason_mask.detach().cpu().numpy() > 0
            raw_risk_prob.append(pre_ans_reason[index])
            answer_reason.append(label_reason.cpu().numpy()[index])
            prediction_reason.append((pre_ans_reason > 0.5)[index])
        else:
            raw_risk_prob.append(pre_ans_reason)
            answer_reason.append(label_reason.cpu().numpy())
            prediction_reason.append(pre_ans_reason>0.5)

    answer = np.concatenate(answer, 0)
    answer_score = np.concatenate(answer_score, 0)
    raw_prediction['state'] = raw_state_prob
    raw_prediction['risk'] = raw_risk_prob
    prediction = np.concatenate(prediction, 0)
    accuracy = accuracy_score(answer, prediction)
    AUC = roc_auc_score(answer, answer_score)
    f1_micro = f1_score(answer, prediction, average='micro')
    f1_macro = f1_score(answer, prediction, average='macro')
    result_one = {}
    result_one['state_acc'] = accuracy
    result_one['state_auc'] = AUC
    print(accuracy)
    print(AUC)
    print(f1_micro)
    print(f1_macro)
    result_one['state_f1_micro'] = f1_micro
    result_one['state_f1_macro'] = f1_macro
    record.write(str(accuracy).encode('utf-8') + b'\n')
    record.write(str(f1_micro).encode('utf-8') + b'\n')
    record.write(str(f1_macro).encode('utf-8') + b'\n')
    print('________________________________')
    record.write('__________________________________'.encode('utf-8') + b'\n')
    answer_reason = np.concatenate(answer_reason, 0)
    prediction_reason = np.concatenate(prediction_reason, 0)
    accuracy = accuracy_score(answer_reason, prediction_reason)
    f1_micro = f1_score(answer_reason, prediction_reason, average='micro')
    f1_macro = f1_score(answer_reason, prediction_reason, average='macro')
    print(accuracy)
    print(f1_micro)
    print(f1_macro)
    record.write(str(accuracy).encode('utf-8') + b'\n')
    record.write(str(f1_micro).encode('utf-8') + b'\n')
    record.write(str(f1_macro).encode('utf-8') + b'\n')
    record.write('__________________________________'.encode('utf-8') + b'\n')
    result_one['avg_acc'] = accuracy
    result_one['avg_f1_micro'] = f1_micro
    result_one['avg_f1_macro'] = f1_macro
    avg_loss = np.array(avg_loss).mean()
    print("TEST LOSS: %f" %avg_loss)
    aucs = []
    nums = []
    raw_risk_prob = np.concatenate(raw_risk_prob, 0)
    for rid, name in zip(range(4), ['fund', 'exp_design', 'recruitment', 'no_expected']):
        cp = raw_risk_prob[:, rid] > 0.5
        ca = answer_reason[:, rid]
        accuracy = accuracy_score(ca, cp)
        f1 = f1_score(ca, cp)
        result_one['Risk_%s_acc' % name] = accuracy
        result_one['Risk_%s_f1' % name] = f1
        roc_auc = roc_auc_score(ca, raw_risk_prob[:, rid])
        aucs.append(roc_auc)
        nums.append(np.sum(ca))
    nums = np.array(nums)
    nums /= np.sum(nums)
    aucs_w = (np.array(aucs) * nums).sum()
    result_one['avg_auc'] = np.mean(aucs)
    result_one['weighted_auc'] = aucs_w
    result_one['raw_record'] = raw_prediction
    model.train()
    return accuracy, f1_micro, f1_macro, avg_loss, result_one

def start():
    try:
        torch.cuda.set_device(config.gpuid)
    except:
        torch.cuda.set_device(0)
    record = open('record.txt', 'wb', buffering=0)
    model, optimizer, optimizer_bert, train_dataloader, val_dataloader, test_dataloader = build(config)
    train(model, optimizer_bert, optimizer, train_dataloader, val_dataloader, test_dataloader, config, record)

def start_test():
    try:
        torch.cuda.set_device(config.gpuid)
    except:
        torch.cuda.set_device(0)
    results = []
    for k in range(5):
        record = open('record.txt', 'wb', buffering=0)
        model, optimizer, optimizer_bert, train_dataloader, val_dataloader, test_dataloader = build(config)
        best_result_one = train(model, optimizer_bert, optimizer, train_dataloader, val_dataloader, test_dataloader, config, record)
        results.append(best_result_one)
    print("state")
    for key_name in results[0].keys():
        if key_name == 'raw_record':
            continue
        values = 0
        for t in results:
            values += t[key_name]
        values /= len(results)
        print("FINAL AVG %s : %f" % (key_name, values))
    pickle.dump(results, open('./results/'+varname(config.model) + '_record_final.pkl', 'wb'))