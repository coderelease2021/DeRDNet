import sys

from baseline.train import start_test as start_test_ori
from baseline.train_WOBert import start_test as start_test_wo_ori
from environment import config
import re
def varname(p):
    m = re.findall(r'\'(.*)\'', str(p))
    if m:
        return m[0]
if __name__ == '__main__':
    model = config.model
    m_name = varname(model)
    if 'BERT' in m_name or 'XLNet' in m_name:
        start_test_ori()
    else:
        start_test_wo_ori()