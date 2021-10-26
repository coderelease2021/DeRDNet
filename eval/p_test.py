import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import pickle
resultsB = pickle.load(open('../results/baseline.Han.BaseModel_record_final.pkl', 'rb'))
resultsO = pickle.load(open('../results/model_folder.dynamic.DynamicDecisionFilterGate_record_b_final(S3_D128_R0.750000LR0.000002).pkl', 'rb'))['results']
for key_name in resultsB[0].keys():
    if key_name == 'raw_record':
        continue
    valuesB = []
    for t in resultsB:
        valuesB.append(t[key_name])
    valuesB = np.array(valuesB)

    valuesO = []
    for t in resultsO:
        valuesO.append(t[key_name])
    valuesO = np.array(valuesO)
    stat_val, p_val = stats.ttest_ind(valuesO, valuesB, equal_var=False)
    print("P_val %s : %f" % (key_name, p_val))