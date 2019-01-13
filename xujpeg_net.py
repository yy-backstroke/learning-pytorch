# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:40:08 2019

@author: DIP16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        kernel=[[[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]],
                [[0.326641,0.135299,-0.135299,-0.326641],[0.326641,0.135299,-0.135299,-0.326641],[0.326641,0.135299,-0.135299,-0.326641],[0.326641,0.135299,-0.135299,-0.326641]],
                [[0.326641,0.326641,0.326641,0.326641],[0.135299,0.135299,0.135299,0.135299],[-0.135299,-0.135299,-0.135299,-0.135299],[-0.326641,-0.326641,-0.326641,-0.326641]],
                [[0.25,-0.25,-0.25,0.25],[0.25,-0.25,-0.25,0.25],[0.25,-0.25,-0.25,0.25],[0.25,-0.25,-0.25,0.25]],
                [[0.426777,0.176777,-0.176777,-0.426777],[0.176777,0.0732233,-0.0732233,-0.176777],[-0.176777,-0.0732233,0.0732233,0.176777],[-0.426777,-0.176777,0.176777,0.426777]],
                [[0.25,0.25,0.25,0.25],[-0.25,-0.25,-0.25,-0.25],[-0.25,-0.25,-0.25,-0.25],[0.25,0.25,0.25,0.25]],
                [[0.135299,-0.326641,0.326641,-0.135299],[0.135299,-0.326641,0.326641,-0.135299],[0.135299,-0.326641,0.326641,-0.135299],[0.135299,-0.326641,0.326641,-0.135299]],
                [[0.326641,-0.326641,-0.326641,0.326641],[0.135299,-0.135299,-0.135299,0.135299],[-0.135299,0.135299,0.135299,-0.135299],[-0.326641,0.326641,0.326641,-0.326641]],
                [[0.326641,0.135299,-0.135299,-0.326641],[-0.326641,-0.135299,0.135299,0.326641],[-0.326641,-0.135299,0.135299,0.326641],[0.326641,0.135299,-0.135299,-0.326641]],
                [[0.135299,0.135299,0.135299,0.135299],[-0.326641,-0.326641,-0.326641,-0.326641],[0.326641,0.326641,0.326641,0.326641],[-0.135299,-0.135299,-0.135299,-0.135299]],
                [[0.176777,-0.426777,0.426777,-0.176777],[0.0732233,-0.176777,0.176777,-0.0732233],[-0.0732233,0.176777,-0.176777,0.0732233],[-0.176777,0.426777,-0.426777,0.176777]],
                [[0.25,-0.25,-0.25,0.25],[-0.25,0.25,0.25,-0.25],[-0.25,0.25,0.25,-0.25],[0.25,-0.25,-0.25,0.25]],
                [[0.176777,0.0732233,-0.0732233,-0.176777],[-0.426777,-0.176777,0.176777,0.426777],[0.426777,0.176777,-0.176777,-0.426777],[-0.176777,-0.0732233,0.0732233,0.176777]],
                [[0.135299,-0.326641,0.326641,-0.135299],[-0.135299,0.326641,-0.326641,0.135299],[-0.135299,0.326641,-0.326641,0.135299],[0.135299,-0.326641,0.326641,-0.135299]],
                [[0.135299,-0.135299,-0.135299,0.135299],[-0.326641,0.326641,0.326641,-0.326641],[0.326641,-0.326641,-0.326641,0.326641],[-0.135299,0.135299,0.135299,-0.135299]],
                [[0.0732233,-0.176777,0.176777,-0.0732233],[-0.176777,0.426777,-0.426777,0.176777],[0.176777,-0.426777,0.426777,-0.176777],[-0.0732233,0.176777,-0.176777,0.0732233]]]
        kernel = torch.FloatTensor(kernel).unsqueeze(1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        #group1
        self.conv1_short = nn.Conv2d(16, 24, kernel_size=3,stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm2d(24)
        self.conv1_1 = nn.Conv2d(16, 12, kernel_size=3,stride=1,padding=1)
        self.bn1_2 = nn.BatchNorm2d(12)
        self.conv1_2 = nn.Conv2d(12, 24, kernel_size=3,stride=2,padding=1)
        self.bn1_3 = nn.BatchNorm2d(24)
        #group2
        self.conv2_1 = nn.Conv2d(24, 24, kernel_size=3,stride=1,padding=1)
        self.bn2_1 = nn.BatchNorm2d(24)
        self.conv2_2 = nn.Conv2d(24, 24, kernel_size=3,stride=1,padding=1)
        self.bn2_2 = nn.BatchNorm2d(24)
        #group3
        self.conv3_short = nn.Conv2d(24, 48, kernel_size=3,stride=2,padding=1)
        self.bn3_1 = nn.BatchNorm2d(48)
        self.conv3_1 = nn.Conv2d(24, 24, kernel_size=3,stride=1,padding=1)
        self.bn3_2 = nn.BatchNorm2d(24)
        self.conv3_2 = nn.Conv2d(24, 48, kernel_size=3,stride=2,padding=1)
        self.bn3_3 = nn.BatchNorm2d(48)
        #group4
        self.conv4_1 = nn.Conv2d(48, 48, kernel_size=3,stride=1,padding=1)
        self.bn4_1 = nn.BatchNorm2d(48)
        self.conv4_2 = nn.Conv2d(48, 48, kernel_size=3,stride=1,padding=1)
        self.bn4_2 = nn.BatchNorm2d(48)
        #group5
        self.conv5_short = nn.Conv2d(48, 96, kernel_size=3,stride=2,padding=1)
        self.bn5_1 = nn.BatchNorm2d(96)
        self.conv5_1 = nn.Conv2d(48, 48, kernel_size=3,stride=1,padding=1)
        self.bn5_2 = nn.BatchNorm2d(48)
        self.conv5_2 = nn.Conv2d(48, 96, kernel_size=3,stride=2,padding=1)
        self.bn5_3 = nn.BatchNorm2d(96)
        
        #group6
        self.conv6_1 = nn.Conv2d(96, 96, kernel_size=3,stride=1,padding=1)
        self.bn6_1 = nn.BatchNorm2d(96)
        self.conv6_2 = nn.Conv2d(96, 96, kernel_size=3,stride=1,padding=1)
        self.bn6_2 = nn.BatchNorm2d(96)
        #group7
        self.conv7_short = nn.Conv2d(96, 192, kernel_size=3,stride=2,padding=1)
        self.bn7_1 = nn.BatchNorm2d(192)
        self.conv7_1 = nn.Conv2d(96, 96, kernel_size=3,stride=1,padding=1)
        self.bn7_2 = nn.BatchNorm2d(96)
        self.conv7_2 = nn.Conv2d(96, 192, kernel_size=3,stride=2,padding=1)
        self.bn7_3 = nn.BatchNorm2d(192)
        #group8
        self.conv8_1 = nn.Conv2d(192, 192, kernel_size=3,stride=1,padding=1)
        self.bn8_1 = nn.BatchNorm2d(192)
        self.conv8_2 = nn.Conv2d(192, 192, kernel_size=3,stride=1,padding=1)
        self.bn8_2 = nn.BatchNorm2d(192)
        #group9
        self.conv9_short = nn.Conv2d(192, 384, kernel_size=3,stride=2,padding=1)
        self.bn9_1 = nn.BatchNorm2d(384)
        self.conv9_1 = nn.Conv2d(192, 192, kernel_size=3,stride=1,padding=1)
        self.bn9_2 = nn.BatchNorm2d(192)
        self.conv9_2 = nn.Conv2d(192, 384, kernel_size=3,stride=2,padding=1)
        self.bn9_3 = nn.BatchNorm2d(384)
        #group10
        self.conv10_1 = nn.Conv2d(384, 384, kernel_size=3,stride=1,padding=1)
        self.bn10_1 = nn.BatchNorm2d(384)
        self.conv10_2 = nn.Conv2d(384, 384, kernel_size=3,stride=1,padding=1)
        self.bn10_2 = nn.BatchNorm2d(384)
        
        self.fc = nn.Linear(384, 2)
    def forward(self, x):
        x-=0.5
        #用固定的DCT4[16*4*4]卷积核
        x = F.conv2d(x,self.weight,padding=1)
        x=torch.abs(x)
        #截断，阈值8
        x=torch.clamp(x,-8/255.,8/255.)
        
        #group1.short:
        x1=self.conv1_short(x)
        x1=self.bn1_1(x1)
        #group1:
        x2=self.conv1_1(x)
        x2=self.bn1_2(x2)
        x2=F.relu(x2)
        x2=self.conv1_2(x2)
        x2=self.bn1_3(x2)
        x=torch.add(x1,x2)
        x=F.relu(x)
        
        #group2:
        x1=self.conv2_1(x)
        x1=self.bn2_1(x1)
        x1=F.relu(x1)
        x1=self.conv2_2(x1)
        x1=self.bn2_2(x1)
        x=torch.add(x,x1)
        x=F.relu(x)
        
        #group3.short:
        x1=self.conv3_short(x)
        x1=self.bn3_1(x1)
        #group3:
        x2=self.conv3_1(x)
        x2=self.bn3_2(x2)
        x2=F.relu(x2)
        x2=self.conv3_2(x2)
        x2=self.bn3_3(x2)
        x=torch.add(x1,x2)
        x=F.relu(x)
        
        #group4:
        x1=self.conv4_1(x)
        x1=self.bn4_1(x1)
        x1=F.relu(x1)
        x1=self.conv4_2(x1)
        x1=self.bn4_2(x1)
        x=torch.add(x,x1)
        x=F.relu(x)
        #group5.short:
        x1=self.conv5_short(x)
        x1=self.bn5_1(x1)
        #group5:
        x2=self.conv5_1(x)
        x2=self.bn5_2(x2)
        x2=F.relu(x2)
        x2=self.conv5_2(x2)
        x2=self.bn5_3(x2)
        x=torch.add(x1,x2)
        x=F.relu(x)
        
        #group6:
        x1=self.conv6_1(x)
        x1=self.bn6_1(x1)
        x1=F.relu(x1)
        x1=self.conv6_2(x1)
        x1=self.bn6_2(x1)
        x=torch.add(x,x1)
        x=F.relu(x)
        #group7.short:
        x1=self.conv7_short(x)
        x1=self.bn7_1(x1)
        #group7:
        x2=self.conv7_1(x)
        x2=self.bn7_2(x2)
        x2=F.relu(x2)
        x2=self.conv7_2(x2)
        x2=self.bn7_3(x2)
        x=torch.add(x1,x2)
        x=F.relu(x)
        
        #group8:
        x1=self.conv8_1(x)
        x1=self.bn8_1(x1)
        x1=F.relu(x1)
        x1=self.conv8_2(x1)
        x1=self.bn8_2(x1)
        x=torch.add(x,x1)
        x=F.relu(x)
        #group9.short:
        x1=self.conv9_short(x)
        x1=self.bn9_1(x1)
        #group9:
        x2=self.conv9_1(x)
        x2=self.bn9_2(x2)
        x2=F.relu(x2)
        x2=self.conv9_2(x2)
        x2=self.bn9_3(x2)
        x=torch.add(x1,x2)
        x=F.relu(x)
        
        #group10:
        x1=self.conv10_1(x)
        x1=self.bn10_1(x1)
        x1=F.relu(x1)
        x1=self.conv10_2(x1)
        x1=self.bn10_2(x1)
        x=torch.add(x,x1)
        x=F.relu(x)
        #global_pool
        x= F.avg_pool2d(x, 16,stride=1) 
        #fc
        x = x.view(-1, 384)
        x = F.relu(self.fc(x))
        return F.log_softmax(x, dim=1)
