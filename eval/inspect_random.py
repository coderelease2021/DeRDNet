import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
result = pickle.load(open('../results/models.BaseModelAttentionHidden_record.pkl', 'rb'))
for round in result:
    temp = []
    for batch in round['state']:
        temp.append(np.argmax(batch, 1))
    prediction = np.concatenate(temp, axis=0)
    answer = round['answer']
    prediction = np.random.randint(0, 2, len(answer))
    accuracy = accuracy_score(answer, prediction)
    print('State acc: %f' %(accuracy))
    answer_reason = np.concatenate(round['answer_reason'], 0)
    prediction_risk = np.concatenate(round['risk'], 0)
    for rid, name in zip(range(4), ['fund', 'exp_design', 'recruitment', 'no_expected']):
        cp = prediction_risk[:, rid] > 0.5
        ca = answer_reason[:, rid]
        cp = np.random.randint(0, 2, len(ca))
        accuracy = accuracy_score(ca, cp)
        print('Risk: %s, acc: %f' %(name, accuracy))
