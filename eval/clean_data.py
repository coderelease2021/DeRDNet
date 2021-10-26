import pickle
import random
raw_data = pickle.load(open('../small_data.pkl', 'rb'))
key_map = {0:1, 1:3, 2:3, 3:3, 4:2, 5:2, 6:1, 7:1, 8:0, 9:2, 10:4}
for i in range(len(raw_data)-1, -1, -1):
    dp = raw_data[i]
    if 'simplified_reason_new' in dp and dp['simplified_reason_new'][10] == 1:
        del raw_data[i]
    if 'simplified_reason_new' in dp:
        simplified_reason = [0 for x in range(4)]
        for kid, label in enumerate(dp['simplified_reason_new']):
            if label > 0 and kid != 10:
                simplified_reason[key_map[kid]] = 1
        dp['simplified_reason_new'] = simplified_reason
pickle.dump(raw_data, open('../beta_data.pkl', 'wb'))
