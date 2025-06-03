import torch.nn as nn
import torch


# building the model
class CNN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1):
    super(CNN, self).__init__()
    self.layers = []
    self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False))
    self.layers.append(nn.BatchNorm2d(out_channels))
    self.layers.append( nn.LeakyReLU(0.1))
    self.layers = nn.Sequential(*self.layers)

    nn.init.kaiming_normal_(self.layers[0].weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
  def forward(self, x):
    return self.layers(x)

class MaxPooling(nn.Module):
  def __init__(self, kernel_size, stride):
    super(MaxPooling, self).__init__()
    self.maxpooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
  def forward(self, x):
    x = self.maxpooling(x)
    return x

class YOLONetwork(nn.Module):
  def __init__(self):
    super(YOLONetwork, self).__init__()
    self.conv1 = CNN(in_channels=3, out_channels=64, kernel_size=7, stride=1)
    self.maxpooling1 = MaxPooling(kernel_size=2, stride=2)

    self.conv2 = CNN(in_channels=64, out_channels=192, kernel_size=3)
    self.maxpooling2 = MaxPooling(kernel_size=2, stride=2)

    self.conv3 = CNN(in_channels=192, out_channels=128, kernel_size=1)

    self.conv4 = CNN(in_channels=128, out_channels=256, kernel_size=3)
    self.conv5 = CNN(in_channels=256, out_channels=256, kernel_size=1)
    self.conv6 = CNN(in_channels=256 , out_channels=512, kernel_size=3)
    self.maxpooling3 = MaxPooling(kernel_size=2, stride=2)

    self.conv7 = CNN(in_channels=512, out_channels=256, kernel_size=1)

    self.conv8 = CNN(in_channels=256, out_channels=512, kernel_size=3)
    self.conv9 = CNN(in_channels=512, out_channels=256, kernel_size=1)
    self.conv10 = CNN(in_channels=256, out_channels=512, kernel_size=3)
    self.conv11 = CNN(in_channels=512, out_channels=256, kernel_size=1)
    self.conv12 = CNN(in_channels=256, out_channels=512, kernel_size=3)
    self.conv13 = CNN(in_channels=512, out_channels=256, kernel_size=1)
    self.conv14 = CNN(in_channels=256, out_channels=512, kernel_size=3)
    self.conv15 = CNN(in_channels=512, out_channels=512, kernel_size=1)
    self.conv16 = CNN(in_channels=512, out_channels=1024, kernel_size=3)
    self.maxpooling4 = MaxPooling(kernel_size=2, stride=2)

    self.conv17 = CNN(in_channels=1024, out_channels=512, kernel_size=1)

    self.conv18 = CNN(in_channels=512, out_channels=1024, kernel_size=3)
    self.conv19 = CNN(in_channels=1024, out_channels=512, kernel_size=1)
    self.conv20 = CNN(in_channels=512, out_channels=1024, kernel_size=3)
    self.conv21 = CNN(in_channels=1024, out_channels=1024, kernel_size=3)
    self.conv22 = CNN(in_channels=1024, out_channels=1024, kernel_size=3, stride=2)

    self.conv23 = CNN(in_channels=1024, out_channels=1024, kernel_size=3)
    self.conv24 = CNN(in_channels=1024, out_channels=1024, kernel_size=3)

    self.flc1 = nn.Linear(in_features=9216, out_features=4096)
    self.flc2 = nn.Linear(in_features=4096, out_features=7*7*30)
    self.activation = nn.LeakyReLU(0.1)
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    x = x.float()
    batch_size = x.shape[0]
    x = self.conv1.forward(x)
    x = self.maxpooling1.forward(x)
    x = self.conv2.forward(x)
    x = self.maxpooling2.forward(x)
    x = self.conv3.forward(x)
    x = self.conv4.forward(x)
    x = self.conv5.forward(x)
    x = self.conv6.forward(x)
    x = self.maxpooling3.forward(x)
    x = self.conv7.forward(x)
    x = self.conv8.forward(x)
    x = self.conv9.forward(x)
    x = self.conv10.forward(x)
    x = self.conv11.forward(x)
    x = self.conv12.forward(x)
    x = self.conv13.forward(x)
    x = self.conv14.forward(x)
    x = self.conv15.forward(x)
    x = self.conv16.forward(x)
    x =  self.maxpooling4.forward(x)
    x = self.conv17.forward(x)
    x = self.conv18.forward(x)
    x = self.conv19.forward(x)
    x = self.conv20.forward(x)
    x = self.conv21.forward(x)
    x = self.conv22.forward(x)
    x = self.conv23.forward(x)
    x = self.conv24.forward(x)
    x = x.reshape(batch_size, -1) # reshape to (batch_size, v)
    x = self.flc1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.flc2(x)
    # x = self.activation(x)
    x = x.reshape(batch_size, 7, 7, 30)
    return x

if __name__=="__main__":
  network = YOLONetwork()
  input = torch.randn((1, 3, 448, 448))
  pediction = network(input)
