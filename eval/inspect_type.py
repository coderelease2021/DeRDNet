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
study_type = {}
for dp in raw_data:
    st = dp['Discrete']['study_type']
    if st in study_type:
        study_type[st] += 1
    else:
        study_type[st] = 1
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
    for key in texts.keys():
        total_text += texts[key]
        try:
            group_text_length[key].append(len(texts[key].split()))
        except:
            group_text_length[key]=[len(texts[key].split())]
    total_text_length.append(len(total_text.split()))


#每个标签占多大，会自动去算百分比
x = [t for t in study_type.values()]
label = [t for t in study_type.keys()]
import matplotlib.pyplot as plt
# 绘制饼图
plt.pie(x,labels=label,autopct='%.0f%%')
plt.show()
print(study_type)

