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
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=5)
        
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        #x = F.conv2d(x,self.weight)
        x = F.avg_pool2d(F.relu(self.conv1(x)),3,stride=2,padding=1)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 3,stride=2,padding=1)
        x = F.avg_pool2d(F.relu(self.conv3(x)), 3,stride=2)
        x = F.avg_pool2d(F.relu(self.conv4(x)), 3,stride=2)
        x = F.avg_pool2d(F.relu(self.conv5(x)), 3,stride=2)
        x = x.view(-1, 256)
        print(self.conv1.weight.data[0])

        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
