from collections import OrderedDict

from torch import nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=3,padding=1)),
            ('relu1', nn.PReLU()),

            ('conv2', nn.Conv2d(32, 32, kernel_size=3,padding=1)),
            ('relu2', nn.PReLU()),
            ('drop2', nn.Dropout(p=0.25)),

            ('conv3', nn.Conv2d(32, 8, kernel_size=3,padding=1)),
            ('relu3', nn.PReLU()),
            ('drop3', nn.Dropout(p=0.7)),
        ]))
        
        self.dense_layer = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(8*7*147, 256)),
            ('relu1',  nn.PReLU()),
            ('drop1',  nn.Dropout(p=0.7)),

            ('dense2', nn.Linear(256, 128)),
            ('relu2',  nn.PReLU()),
            ('drop2',  nn.Dropout(p=0.6)),

            ('dense3', nn.Linear(128, 2)),
        ]))
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 8*7*147) 
        x = self.dense_layer(x)
        return x
