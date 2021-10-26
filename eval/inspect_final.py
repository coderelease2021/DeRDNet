import pickle
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

results = pickle.load(open('../results/model_folder.dynamic.DynamicDecisionGate_record_b_final(S3_D128_R0.750000LR0.000002).pkl', 'rb'))
if 'config' in results:
    results = results['results']
result_ = pickle.load(open('../small_data_clean.pkl', 'rb'))
result_ = result_[len(result_)-1000:]
answer = []
for case in result_:
    if case['state'] in ['Completed', 'Available', 'Approved for marketing']:
        state = 0
    else:
        state = 1
    answer.append(state)
answer = np.array(answer)
for key_name in results[0].keys():
    if key_name == 'raw_record':
        continue
    values = []
    for t in results:
        values.append(t[key_name])
    values = np.array(values)
    print("FINAL AVG %s : %f" % (key_name, values.mean()))
    print("FINAL AVG STD %s : %f" % (key_name, values.std()))

AUCs = []
for round in results:
    temp = []
    for batch in round['raw_record']['state']:
        temp.append(batch[:, 1])
    prediction_score = np.concatenate(temp, axis=0)
    accuracy = roc_auc_score(answer, prediction_score)
    print('state auc: %f' %(accuracy))
    AUCs.append(accuracy)
AUCs = np.array(AUCs)
print(AUCs.mean())
print(AUCs.std())