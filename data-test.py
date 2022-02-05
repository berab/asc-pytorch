import numpy as np
import pandas as pd
import pickle

filepath = './park-barcelona-241-7162-a.logmel'
feat_mtx = []
with open(filepath, 'rb') as f:
   temp=pickle.load(f, encoding='latin1')
   feat_mtx.append(temp['feat_data'])

feat_mtx = np.array(feat_mtx)

import pdb; pdb.set_trace()