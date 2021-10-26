import pickle
import random

import numpy as np

raw_data = pickle.load(open('../small_data_clean.pkl', 'rb'))
print(len(raw_data))
c_count = 0
simpls = []
total_text_length = []
group_text_length = {}
risks = []
key_map = {0:1, 1:3, 2:3, 3:3, 4:2, 5:2, 6:1, 7:1, 8:0, 9:2, 10:4}
for dp in raw_data:
    total_text = ''
    if dp['state'] == 'Completed':
        c_count += 1
    else:
        simplified_reason = [0 for x in range(4)]
        for kid, label in enumerate(dp['simplified_reason_new']):
            if label > 0 and kid != 10:
                simplified_reason[key_map[kid]] = 1
        simplified_reason = np.array(simplified_reason)
        risks.append(simplified_reason)

    texts = dp['Text']
    print(dp['CID'])
    print(texts['overall_official'])
    for key in texts.keys():
        total_text += texts[key]
        try:
            group_text_length[key].append(len(texts[key].split()))
        except:
            group_text_length[key]=[len(texts[key].split())]
    total_text_length.append(len(total_text.split()))
print('Completed')
print(c_count)
print('avg len')
print(np.mean(total_text_length))
print('sec avg len')
for key in group_text_length.keys():
    print(key)
    print(np.mean(group_text_length[key]))
risks = np.array(risks)
risks = risks.sum(axis=0)
risks_ratio = risks/np.sum(risks)
print(risks_ratio)
