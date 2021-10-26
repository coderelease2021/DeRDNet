import pickle
import numpy as np
raw_data = pickle.load(open('../small_data_clean.pkl', 'rb'))
key_map = {0:1, 1:3, 2:3, 3:3, 4:2, 5:2, 6:1, 7:1, 8:0, 9:2, 10:4}
for did, dp in enumerate(raw_data):
    total_text = ''
    if dp['state'] == 'Completed':
        continue
    else:
        simplified_reason = [0 for x in range(4)]
        for kid, label in enumerate(dp['simplified_reason_new']):
            if label > 0 and kid != 10:
                simplified_reason[key_map[kid]] = 1
        simplified_reason = np.array(simplified_reason)
        raw_data[did]['simplified_reason_new'] = simplified_reason
pickle.dump(raw_data, open('../release_data_clean.pkl', 'wb'))
pickle.dump(raw_data[0:200], open('../release_data_sample.pkl', 'wb'))