from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, strides=1, learn_bn=True, wd=1e-4, use_relu=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels, affine=learn_bn)
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides)

    def forward(self, x):
        x = self.bn(x)
        if self.use_relu:
            x = F.relu(x)
        x = self.conv(x)
        return x

x = torch.ones(1, 1, 128, 128)
net = ResnetLayer()
print(net(x).shape)