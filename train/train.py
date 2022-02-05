import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import sys
sys.path.append('..')
sys.path.append('../models')
#from utils import *
#from funcs import *

from mobnet import ModelMobnet
from csv_test import load_labels

#dummies
csv_path = '../data_2020/evaluation_setup/fold1_train_small.csv'
labels = load_labels(csv_path)[0:100]
dummy_input = torch.ones(labels.shape[0], 6, 128, 461)

net = ModelMobnet(num_classes=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
labels = torch.from_numpy(labels).type(torch.LongTensor)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0,10):
        # get the inputs; data is a list of [inputs, labels]
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = net(dummy_input).reshape(100,3)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
            

print('Finished Training')