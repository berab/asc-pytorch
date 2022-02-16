import os
import argparse
from tabnanny import verbose
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from training_functions import mixup_data, mixup_criterion
from torchaudio import transforms

import sys

sys.path.append('..')
sys.path.append('../models')
from utils import *
from cnn import Cnn

data_path = '../data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
feat_path = '../features/logmel128_scaled_d_dd/'
experiments = 'exp_mobnet'

if not os.path.exists(experiments):
    os.makedirs(experiments)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_freq_bin = 128
num_audio_channels = 2
num_classes = 3
batch_size=64
num_epochs=10
sample_num = len(open(train_csv, 'r').readlines()) - 1
alpha = 0.4
mx_lr = 0.001

X_train, y_train = load_data_2020(feat_path, train_csv, num_freq_bin, 'logmel')
print('Training data unpickled!')
X_train = np.transpose(X_train,(0,3,1,2)) # need to change channel last to channel one

X_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
print('Validation data unpicked!')
X_val = np.transpose(X_val,(0,3,1,2)) 

trainloader = torch.utils.data.DataLoader([[X_train[i], y_train[i]] for i in range(len(y_train))], 
                                            batch_size=batch_size, shuffle=True, num_workers=2) 
validloader = torch.utils.data.DataLoader([[X_val[i], y_val[i]] for i in range(len(y_val))], 
                                            batch_size=batch_size, num_workers=2) 

net = Cnn()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=mx_lr, momentum=0.9)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=mx_lr*1e-4, verbose=False)

freq_mask = transforms.FrequencyMasking(13,True).to(device)
time_mask = transforms.TimeMasking(40,True).to(device)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    train_loss = 0.0
    net.train()
    #dummy input
    
    for i, data in enumerate(trainloader):

        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)

       
        inputs = time_mask(freq_mask(inputs)) # masking
        inputs = time_mask(freq_mask(inputs))

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # if using warm restart, then update after each batch iteration

        # print statistics
        train_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[Epoch, Batch]: [{epoch + 1}, {i + 1:5d}] \t\t Training loss: {train_loss / 100:.3f}')
            #print(lr_scheduler.get_last_lr())
            train_loss = 0.0

    valid_loss = 0.0
    correct = 0
    total = 0
    net.eval()

    for i, data in enumerate(validloader):
        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum()
    
    acc = 100.*correct/total 
    print(f'Epoch {epoch+1} \t\t Validation Loss: {valid_loss / len(validloader)} \t\t Accuracy: {acc} %')

PATH = './mobilenet.pth'
torch.save(net.state_dict(), PATH)
print('Finished Training')  
