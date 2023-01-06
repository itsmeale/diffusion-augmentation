import torch
from torch import nn


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3)
        self.pool = nn.AvgPool2d(10)

        self.idrop = nn.Dropout2d(.2)
        self.hdrop = nn.Dropout1d(.2)

        self.fc1 = nn.Linear(11664, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def conv_layer(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        return x
    
    def fc_layer(self, x):
        x = self.fc1(x)
        x = self.hdrop(x)
        x = self.sig(x)

        x = self.fc2(x)        
        x = self.sig(x)
        return x
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x
