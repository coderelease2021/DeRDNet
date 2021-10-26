import pickle
import matplotlib.pyplot as plt
import numpy as np

depth_list =[]
depth_list_risk =[]
for depth in [1,2,3,4]:
    record = pickle.load(open('../results/model_folder.dynamic.DynamicDecisionFilterGate_record_b_final(S%d_D128_R0.750000LR0.000002).pkl' %depth, 'rb'))
    results = record['results']
    state_auc = []
    risk_w_auc = []
    for round in results:
        state_auc.append(round['state_auc'])
        risk_w_auc.append(round['weighted_auc'])
    depth_list.append(state_auc)
    depth_list_risk.append(risk_w_auc)
depth_list = np.array(depth_list).T
depth_list_risk = np.array(depth_list_risk).T
plt.subplot(121)
names = ['2', '3', '4', '5']
x = range(1, len(names)+1)
y1 = [0.7597, 0.7597, 0.7613, 0.7615]
y2 = [0.5630, 0.5673, 0.5666, 0.5540]
plt.boxplot(depth_list)
#plt.xticks(x, names)
plt.xlabel("Depth")
plt.ylabel("Statue AUC")
plt.plot(x, y1, marker='o', color='red', label=u'Mean AUC')
plt.legend()
plt.xticks(x, names)
plt.subplot(122)
plt.boxplot(depth_list_risk)
#plt.xticks(x, names)
plt.xlabel("Depth")
plt.ylabel("Micro Risk AUC")
plt.plot(x, y2, marker='*', color='blue', label=u'Mean AUC')
plt.legend()
plt.xticks(x, names)
plt.tight_layout()
plt.show()