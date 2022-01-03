import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
#import random
import time
#from scipy.stats import multivariate_normal
#from scipy.stats import norm
#import math
#import os
class RollingWindowConv(nn.Module):
    def __init__(self,input_length,num_features,num_output,rolling_size,last_activation,stride1=1,stride2=1):
        super(RollingWindowConv, self).__init__()
        self.num_features = num_features
        self.conv1d = nn.Conv1d(
            in_channels = num_features, 
            out_channels = num_features+1, 
            kernel_size = rolling_size,
            stride = stride1 
        )
        # output = (batch,out_channel, int(1+(window_size-rolling_size)/stride) )
        self.bn1d = nn.BatchNorm1d(self.num_features+1)
        self.conv1_2_reshape = lambda x: torch.unsqueeze(x,1)
        self.conv2d_out_channel = 2
        self.conv2d = nn.Conv2d(
            1,
            self.conv2d_out_channel,
            kernel_size = (num_features+1,rolling_size),
            stride = stride2
        )
        self.bn2d = nn.BatchNorm2d(self.conv2d_out_channel)
        l1_out = int((input_length-rolling_size)/stride1+1)
        l2_out = int((l1_out-rolling_size)/stride2+1)
        # output
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv2d_out_channel*1*l2_out,int(l2_out/2)),
            nn.Sigmoid(),
            nn.Linear(int(l2_out/2),num_output),
        )
        self.lastAct = last_activation

    def forward(self,x):
        #out1 = nn.ReLU()(self.conv1d(x))
        out1 = self.bn1d(nn.ReLU()(self.conv1d(x)))
        out2 = self.conv1_2_reshape(out1)
        #out3 = nn.ReLU()(self.conv2d(out2))
        out3 = self.bn2d(nn.ReLU()(self.conv2d(out2)))
        out4 = self.fc(out3)
        return self.lastAct(out4)


class STDConvModel(nn.Module):
    def __init__(self,input_length,num_features,rolling_size,stride1=1,stride2=1):
        super(STDConvModel, self).__init__()
        self.num_features = num_features
        self.conv1d = nn.Conv1d(
            in_channels = num_features, 
            out_channels = num_features+1, 
            kernel_size = rolling_size,
            stride = stride1 
        )
        # output = (batch,out_channel, int(1+(window_size-rolling_size)/stride) )
        self.bn1d = nn.BatchNorm1d(self.num_features+1)
        self.conv1_2_reshape = lambda x: torch.unsqueeze(x,1)
        self.conv2d_out_channel = 2
        self.conv2d = nn.Conv2d(
            1,
            self.conv2d_out_channel,
            kernel_size = (num_features+1,rolling_size),
            stride = stride2
        )
        self.bn2d = nn.BatchNorm2d(self.conv2d_out_channel)
        l1_out = int((input_length-rolling_size)/stride1+1)
        l2_out = int((l1_out-rolling_size)/stride2+1)
        # output
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv2d_out_channel*1*l2_out,int(l2_out/2)),
            nn.Sigmoid(),
            nn.Linear(int(l2_out/2),1),
            nn.Sigmoid()
        )
    def forward(self,x):
        #out1 = nn.ReLU()(self.conv1d(x))
        out1 = self.bn1d(nn.ReLU()(self.conv1d(x)))
        out2 = self.conv1_2_reshape(out1)
        #out3 = nn.ReLU()(self.conv2d(out2))
        out3 = self.bn2d(nn.ReLU()(self.conv2d(out2)))
        out4 = self.fc(out3)
        return out4

class FC(nn.Module):
    def __init__(self,layer_shapes,activation = nn.Sigmoid, out_act = nn.Sigmoid):
        super().__init__()
        self.topology = layer_shapes
        self.layerslist = []
        for i,x in enumerate(self.topology):
            if i != (len(self.topology)-1):
                self.layerslist.append(nn.Linear(*x))
                self.layerslist.append(activation())
            else:
                self.layerslist.append(nn.Linear(*x))
                self.layerslist.append(out_act())
        self.nn = nn.Sequential(*self.layerslist)

    def forward(self,x):
        return self.nn(x)
    def encode(self):
        return torch.cat([x.flatten() for x in self.parameters()])
    def decode(self,w):
        cumidx = 0
        for p in self.parameters():
            nneurons = torch.numel(p)
            p.data = w[cumidx:cumidx+nneurons].reshape(p.data.shape)
            cumidx += nneurons