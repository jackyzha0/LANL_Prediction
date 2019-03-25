import numpy as np
import pandas as pd
from datetime import datetime

CSVDIR = "data/train.csv"
fix_len = 150000
DBLEN = 6000000
# lim = DBLEN - fix_len
lim = 150000

print('Loading dataset...',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
out = pd.read_csv(CSVDIR, header=0, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows=2*fix_len)

print('Done!',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def get_example():
    ind = np.random.randint(0,lim)
    return out['acoustic_data'][ind:ind+fix_len], out['time_to_failure'][ind:ind+fix_len]

def get_batch(batchsize):
    return np.array([get_example() for x in range(batchsize)])
