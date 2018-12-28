# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:12:20 2018

@author: DIP16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        kernel=[[-1/12., 2/12., -2/12., 2/12., -1/12.],
        [2/12.,-6/12., 8/12., -6/12., 2/12.],
        [-2/12.,8/12., -12/12., 8/12., -2/12.],
        [2/12.,-6/12., 8/12., -6/12., 2/12.],
        [-1/12.,2/12., -2/12., 2/12., -1/12.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5,padding=4)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.conv2d(x,self.weight)
        #group1:
        x=self.conv1(x)
        x=torch.abs(x)
        x=self.bn1(x)
        x=F.tanh(x)
        x= F.avg_pool2d(x, 5,stride=2,padding=2)
        #group2:
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.tanh(x)
        x= F.avg_pool2d(x, 5,stride=2,padding=2)
        #group3:
        x=self.conv3(x)
        x=self.bn3(x)
        x=F.relu(x)
        x= F.avg_pool2d(x, 5,stride=2,padding=2)
        #group4:
        x=self.conv4(x)
        x=self.bn4(x)
        x=F.relu(x)
        x= F.avg_pool2d(x, 5,stride=2,padding=2)     
        #group5:
        x=self.conv5(x)
        x=self.bn5(x)
        x=F.relu(x)
        x= F.avg_pool2d(x, 32,stride=1)   
 
        x = x.view(-1, 128)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
