# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet, resnet

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet121, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet121.classifier = None
            self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet121.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out

class DenseNet161(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet161, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet161 = densenet.densenet161(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet161.classifier.in_features
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet161.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet161.classifier = None
            self.densenet161.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet161.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet161.features(x)
        out = F.relu(features, inplace=True)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet161.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet161.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet161.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet161.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out
    
# ------------------------------------------thêm backbone mới ResNet101---------------------------------
class ResNet101(nn.Module):
    """Model modified."""
    def __init__(self, out_size, mode, drop_rate=0):
        super(ResNet101, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.resnet101 = resnet.resnet101(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.resnet101.fc.in_features
        
        if mode in ('U-Ones', 'U-Zeros'):
            self.resnet101.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass',):
            self.resnet101.classifier = None
            self.resnet101.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.resnet101.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.resnet101.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.resnet101.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.resnet101.conv1(x)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)

        # Perform adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the tensor before passing to fully connected layer
        x = torch.flatten(x, 1)

        if self.drop_rate > 0:
            x = self.drop_layer(x)

        self.activations = x
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.resnet101.classifier(x)
        elif self.mode in ('U-MultiClass',):
            n_batch = x.size(0)
            out_0 = self.resnet101.Linear_0(x).view(n_batch, 1, -1)
            out_1 = self.resnet101.Linear_1(x).view(n_batch, 1, -1)
            out_u = self.resnet101.Linear_u(x).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out
    
# ------------------------------------------thêm backbone mới ResNet101---------------------------------



