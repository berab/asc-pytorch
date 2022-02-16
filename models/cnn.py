import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(6, 16, 4, stride=1, padding=1)
      self.bn1 = nn.BatchNorm2d(16)
      self.maxpool1 = nn.MaxPool2d(6)
      self.dropout = nn.Dropout(p=0.3)
      self.conv2 = nn.Conv2d(16, 24, 16)
      self.bn2 = nn.BatchNorm2d(24)
      self.maxpool2 = nn.MaxPool2d(6)
      self.flatten = nn.Flatten()
      self.dense1 = nn.Linear(240, 50)
      self.dense2 = nn.Linear(50, 3)

   def forward(self, x):
      x = self.conv1(x)
      x = F.relu(self.bn1(x))
      x = self.maxpool1(x)
      x = self.dropout(x)
      x = self.conv2(x)
      x = F.relu(self.bn2(x))
      x = self.maxpool2(x)
      x = self.dropout(x)
      x = self.flatten(x)
      x = self.dense1(x)
      x = self.dropout(x)
      x = self.dense2(x)
      
      return x
      
# net = Cnn()
# x = torch.ones(3, 6, 128, 461)
# x = net(x)
# print(x.shape)