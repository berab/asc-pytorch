import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import sys

sys.path.append('..')
sys.path.append('../models')
from utils import *
#from funcs import *
from mobnet import ModelMobnet
from csv_test import load_labels

data_path = '../data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
feat_path = 'features/logmel128_scaled_d_dd/'
experiments = 'exp_mobnet'

if not os.path.exists(experiments):
    os.makedirs(experiments)

num_audio_channels = 2
num_classes = 3
batch_size=32
num_epochs=2
num_freq_bin=128
sample_num = len(open(train_csv, 'r').readlines()) - 1

data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val = keras.utils.to_categorical(y_val, num_classes)

