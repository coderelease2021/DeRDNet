import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
result = pickle.load(open('../results/models.BaseModelAttentionPlusFilter_record.pkl', 'rb'))
result_ = pickle.load(open('../results/models.BaseModel_record.pkl', 'rb'))
temp = result_[0]
answer = temp['answer']
answer_reason = np.concatenate(temp['answer_reason'], 0)
for round in result:
    temp = []
    for batch in round['state']:
        temp.append(np.argmax(batch, 1))
    prediction = np.concatenate(temp, axis=0)
    accuracy = accuracy_score(answer, prediction)
    print('state acc: %f' %(accuracy))
    prediction_risk = np.concatenate(round['risk'], 0)
    aucs = []
    nums = []
    for rid, name in zip(range(4), ['fund', 'exp_design', 'recruitment', 'no_expected']):
        cp = prediction_risk[:, rid] > 0.5
        ca = answer_reason[:, rid]
        accuracy = accuracy_score(ca, cp)
        f1 = f1_score(ca, cp)
        roc_auc = roc_auc_score(ca, prediction_risk[:, rid])
        print('Risk: %s, acc: %f' %(name, accuracy))
        print('Risk: %s, f1: %f' % (name, f1))
        print('Risk: %s, roc_auc: %f' % (name, roc_auc))
        print('__________________________________________')
        aucs.append(roc_auc)
        nums.append(np.sum(ca))
    nums = np.array(nums)
    nums /= np.sum(nums)
    aucs_w = (np.array(aucs) * nums).sum()
    avg_acc = accuracy_score(answer_reason, prediction_risk>0,5)
    avg_f1_mi = f1_score(answer_reason, prediction_risk>0.5, average='micro')
    avg_f1_ma = f1_score(answer_reason, prediction_risk > 0.5, average='macro')
    print('Risk: average acc: %f' % (avg_acc))
    print('Risk: average f1 micro: %f' % (avg_f1_mi))
    print('Risk: average f1 macro: %f' % (avg_f1_ma))
    print('Risk: average auc: %f' % (np.mean(aucs)))
    print('Risk: weight auc: %f' % (aucs_w))
    print('++++++++++++++++++++++++++++++++++++++++++')

print(nums)

