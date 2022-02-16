import numpy as np
import pandas as pd
import pickle


def run():
   filepath = './park-barcelona-241-7162-a.logmel'
   feat_mtx = []
   with open(filepath, 'rb') as f:
      temp=pickle.load(f, encoding='latin1')
      feat_mtx.append(temp['feat_data'])

   feat_mtx = np.array(feat_mtx)

def data_generation(batch_ids, X_train, y_train):
      _, c, h, w = X_train.shape
      alpha = 0.4
      batch_size=20
      swap_inds = [1, 0, 3, 2, 5, 4]
      
      l = np.random.beta(alpha, alpha, batch_size)
      X_l = l.reshape(batch_size, 1, 1, 1)
      y_l = l.reshape(batch_size, 1)

      X1 = X_train[batch_ids[:batch_size]]
      X2 = X_train[batch_ids[batch_size:]]
      
      for j in range(X1.shape[0]):

         # spectrum augment
         for c in range(X1.shape[3]):
               X1[j, :, :, c] = frequency_masking(X1[j, :, :, c])
               X1[j, :, :, c] = time_masking(X1[j, :, :, c])
               X2[j, :, :, c] = frequency_masking(X2[j, :, :, c])
               X2[j, :, :, c] = time_masking(X2[j, :, :, c])

         # random channel confusion
         if X1.shape[-1]==6:
               if np.random.randint(2) == 1:
                  X1[j, :, :, :] = X1[j:j+1, :, :, swap_inds]
               if np.random.randint(2) == 1:
                  X2[j, :, :, :] = X2[j:j+1, :, :, swap_inds]
      
      # mixup
      X = X1 * X_l + X2 * (1.0 - X_l)

      if isinstance(y_train, list):
         y = []

         for y_train_ in y_train:
               y1 = y_train_[batch_ids[:batch_size]]
               y2 = y_train_[batch_ids[batch_size:]]
               y.append(y1 * y_l + y2 * (1.0 - y_l))
      else:
         y1 = y_train[batch_ids[:batch_size]]
         y2 = y_train[batch_ids[batch_size:]]
         y = y1 * y_l + y2 * (1.0 - y_l)

      return X, y