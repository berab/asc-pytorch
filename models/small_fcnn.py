import torch
from torch import nn
from torch.nn.functional import F

class ResnetLayer(nn.Module):
   def __init__(self, in_channels, out_channels=16, kernel_size=3, stride=1, learn_bn=True, use_relu=True, bias=False):
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
      nn.init.normal_(self.conv.weight)
      self.bn = nn.BatchNorm2d(out_channels,affine=learn_bn)
      self.use_relu = use_relu

   def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      if self.use_relu:
         x = F.relu(x)

      return x

class conv_layer1(nn.Module):
   def __init__(self, in_channels, num_channels=6, num_filters=14, learn_bn=True, use_relu=True):
      super().__init__()
      kernel_size1 = (5, 5)
      kernel_size2 = (3, 3)
      strides1 = (2, 2)
      strides2 = (1, 1)
      self.use_relu = use_relu
      self.bn1 = nn.BatchNorm2d(in_channels, affine=learn_bn)
      self.zeropad1 = nn.ZeroPad2d(2)
      self.conv1 = nn.Conv2d(in_channels, num_channels*num_filters, kernel_size1, strides1, padding='valid',
                              bias=False)
      nn.init.normal_(self.conv1.weight)
      self.bn2 = nn.BatchNorm2d(num_channels*num_filters, affine=learn_bn)
      self.zeropad2 = nn.ZeroPad2d(1)
      self.conv2 = nn.Conv2d(num_filters*num_channels, num_filters*num_channels, kernel_size2, strides2,
                              padding='valid', bias=False)
      nn.init.normal_(self.conv2.weight)
      self.maxpool = nn.MaxPool2d(kernel_size=2, padding='valid')

   def forward(self, x):
      x = self.bn(x)
      x = self.zeropad1(self.conv1(x))
      x = self.bn2(x)
      if self.use_relu:
         x = F.relu(x)

      x = self.bn2(self.conv2(x))
      if self.use_relu:
         x = F.relu(x)
      x = self.maxpool(x)
      return x

class conv_layer2(nn.Module):
   def __init__(self, in_channels, num_channels=6, num_filters=28, learn_bn=True, use_relu=True):
      super().__init__()
      kernel_size = (3, 3)
      stride = 1
      self.use_relu = use_relu

      self.zeropad = nn.ZeroPad2d(1)
      self.conv1 = nn.Conv2d(in_channels, num_channels*num_filters, kernel_size, stride, padding='valid',
                              bias=False)
      nn.init.normal_(self.conv1.weight)
      self.bn = nn.BatchNorm2d(num_channels*num_filters, affine=learn_bn)
      
      self.conv2 = nn.Conv2d(num_filters*num_channels, num_filters*num_channels, kernel_size, stride,
                              padding='valid', bias=False)
      nn.init.normal_(self.conv2.weight)
      self.maxpool = nn.MaxPool2d(kernel_size=2, padding='valid')

   def forward(self, x):
      x = self.zeropad(self.conv1(x))
      x = self.bn(x)
      if self.use_relu:
         x = F.relu(x)
      x = self.zeropad(x)
      x = self.bn(self.conv2(x))
      if self.use_relu:
         x = F.relu(x)
      x = self.maxpool(x)
      return x

class conv_layer3(nn.Module):
   def __init__(self, in_channels, num_channels=6, num_filters=28, learn_bn=True, use_relu=True):
      super().__init__()
      kernel_size = (3, 3)
      stride = 1
      self.use_relu = use_relu

      self.zeropad = nn.ZeroPad2d(1)
      self.conv1 = nn.Conv2d(in_channels, num_channels*num_filters, kernel_size, stride, padding='valid',
                              bias=False)
      nn.init.normal_(self.conv1.weight)
      self.bn = nn.BatchNorm2d(num_channels*num_filters, affine=learn_bn)
      self.dropout = nn.Dropout(0.3)
      self.conv2 = nn.Conv2d(num_filters*num_channels, num_filters*num_channels, kernel_size, stride,
                              padding='valid', bias=False)
      nn.init.normal_(self.conv2.weight)
      self.maxpool = nn.MaxPool2d(kernel_size=2, padding='valid')

   def forward(self, x):
      x = self.zeropad(self.conv1(x))
      x = self.bn(x)
      if self.use_relu:
         x = F.relu(x)
      x = self.dropout(x)
      x = self.zeropad(x)
      x = self.bn(self.conv2(x))
      if self.use_relu:
         x = F.relu(x)
      x = self.maxpool(x)
      return x

class channel_attention(nn.Module): #check dims for nn linear and cont
   def __init__(self, in_channels, ratio=8):
      super().__init__()

      self.fc1 = nn.Linear(in_channels, in_channels//ratio,
                           bias=True
                           )
      nn.init.normal_(self.fc1.bias)
      self.fc2 = nn.Linear(in_channels//ratio, in_channels,
                           bias=True
                           )
      nn.init.normal_(self.fc2.bias)
      
   def forward(self, x):
      inputs = x
      x = x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) #globalavg2d trick
      x = self.fc1(x)
      x = self.fc2(x)
      y, _  = torch.max(inputs, (1,2)) #globalmaxpool?
      y = self.fc1(y)
      y = self.fc2(y)
      x = x + y
      x = F.sigmoid(x)
      return torch.multiply(inputs,x)

class ModelFcnn(nn.Module):
   def __init__(self, num_classes, in_channels=6, num_filters=[24,48,96]):
      super().__init__()
      self.conv_block1 = conv_layer1(in_channels=in_channels,
                                     num_channels=in_channels,
                                     num_filters=num_filters[0],
                                     learn_bn=True,
                                     use_relu=True)

      self.conv_block2 = conv_layer1(in_channels=num_filters[0]*in_channels,
                                     num_channels=in_channels,
                                     num_filters=num_filters[1],
                                     learn_bn=True,
                                     use_relu=True)

      self.conv_block3 = conv_layer1(in_channels=num_filters[1]*in_channels,
                                     num_channels=in_channels,
                                     num_filters=num_filters[2],
                                     learn_bn=True,
                                     use_relu=True)

      self.output_path = ResnetLayer(in_channels=num_filters[2]*in_channels,
                                     out_channels=num_classes,
                                     stride=1,
                                     kernel_size=1,
                                     learn_bn=False,
                                     use_relu=True)

      self.bn = nn.BatchNorm2d(num_classes, affine=False)
      self.channel_attention = channel_attention(in_channels=num_classes, ratio=2)
      
   def forward(self, x):
      x = self.conv_block1(x)
      x = self.conv_block2(x)
      x = self.conv_block3(x)
      x = self.ResnetLayer(x)
      x = self.channel_attention(x)
      x = x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True) #globalavg2d trick
      x = self.softmax(x)