import numpy as np
import pandas as pd

CSVDIR = "data/train.csv"
fix_len = 150000
DBLEN = 6000000
lim = DBLEN - fix_len

out = pd.read_csv(CSVDIR, header=0, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(out['acoustic_data'].shape)
print(out['time_to_failure'].shape)

print(out['acoustic_data'][0],out['time_to_failure'][0])

# def get_example():
#     ind = np.random.randint(0,lim)
