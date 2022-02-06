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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_freq_bin = 128
num_audio_channels = 2
num_classes = 3
batch_size=32
num_epochs=2
sample_num = len(open(train_csv, 'r').readlines()) - 1

# X_train, y_train = load_data_2020(feat_path, train_csv, num_freq_bin, 'logmel')
# X_train = np.transpose(X_train,(0,3,1,2)) # need to change channel last to channel one

X_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
X_val = np.transpose(X_val,(0,3,1,2)) 

# trainloader = torch.utils.data.DataLoader([[X_train[i], y_train[i]] for i in range(len(y_train))], 
#                                             batch_size=batch_size, shuffle=True, num_workers=2) 
validloader = torch.utils.data.DataLoader([[X_val[i], y_val[i]] for i in range(len(y_val))], 
                                            batch_size=batch_size, num_workers=2) 

net = ModelMobnet(num_classes, in_channels=num_audio_channels*3, num_channels=24)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    train_loss = 0.0
    net.train()
    #dummy input

    x = np.random.random((2,3,128,461))
    y = np.random.random((2,3,1,1))
    trainloader = torch.utils.data.DataLoader([[x[i], y[i]] for i in range(len(y))], 
                                            batch_size=batch_size, shuffle=True, num_workers=2) 
    for i, data in enumerate(trainloader):

        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')
            train_loss = 0.0

    valid_loss = 0.0
    net.eval()
    for i, data in enumerate(validloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.type(torch.LongTensor)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        valid_loss = loss.item()*data.size(0)
    
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')

print('Finished Training')  


def data_generation(batch_ids, X_train, y_train):
    _, h, w, c = X_train.shape
    l = np.random.beta(self.alpha, self.alpha, self.batch_size)
    X_l = l.reshape(self.batch_size, 1, 1, 1)
    y_l = l.reshape(self.batch_size, 1)

    X1 = X_train[batch_ids[:self.batch_size]]
    X2 = X_train[batch_ids[self.batch_size:]]
    
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
                X1[j, :, :, :] = X1[j:j+1, :, :, self.swap_inds]
            if np.random.randint(2) == 1:
                X2[j, :, :, :] = X2[j:j+1, :, :, self.swap_inds]
    
    # mixup
    X = X1 * X_l + X2 * (1.0 - X_l)

    if isinstance(y_train, list):
        y = []

        for y_train_ in y_train:
            y1 = y_train_[batch_ids[:self.batch_size]]
            y2 = y_train_[batch_ids[self.batch_size:]]
            y.append(y1 * y_l + y2 * (1.0 - y_l))
    else:
        y1 = y_train[batch_ids[:self.batch_size]]
        y2 = y_train[batch_ids[self.batch_size:]]
        y = y1 * y_l + y2 * (1.0 - y_l)

    return X, y
