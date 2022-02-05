import numpy as np
import pandas as pd
import pickle

def load_labels(csv_path):
   
   with open(csv_path, 'r') as text_file:
      lines = text_file.read().split('\n')
      for idx, elem in enumerate(lines):
         lines[idx] = lines[idx].split('\t')
         lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

      # remove first line
      lines = lines[1:]
      lines = [elem for elem in lines if elem != ['']]
      for idx, elem in enumerate(lines):
         lines[idx][-1] = lines[idx][-1].split('\r')[0]
      label_info = np.array(lines)

      data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
      ClassNames = np.unique(data_df['scene_label'])
      labels = data_df['scene_label'].astype('category').cat.codes.values
      return labels

