# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:37:11 2018

@author: DIP16
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from qian_net import Net as qian_net
from xu_net import Net as xu_net

from imgFolder import ImageFolder
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='wd',
                    help='weight_decay (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def show_batch(imgs):
    grid = utils.make_grid(imgs,nrow=5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_path = "../../database/train"
test_path = "../../database/test"

train_data= ImageFolder(train_path, 
            transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
test_data=ImageFolder(test_path, transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
#transforms.Normalize((0.1307,), (0.3081,))
train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.test_batch_size, shuffle=True, **kwargs)
#transforms.TenCrop(256),transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))


model = xu_net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, momentum=args.momentum)


#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
#                            momentum=args.momentum,
#                            weight_decay=args.weight_decay)        
    # step4: meters
#    loss_meter = meter.AverageValueMeter()
#    confusion_matrix = meter.ConfusionMeter(2)
#    previous_loss = 1e100


def train(epoch):
    model.train()
    for batch_idx, ((data1, target1),(data2, target2)) in enumerate(train_loader):
        data=torch.cat([data1,data2],0)
        target=torch.cat([target1,target2],0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
#        bs, ncrops, c, h, w = data.size()
#        data=data.view(-1, c, h, w)
        output = model(data)
#        output_avg = output.view(bs, ncrops, -1).mean(1)
        loss = F.nll_loss(output, target)
#        print(output,output_avg,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % (10*args.log_interval) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx/10) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for ((data1, target1),(data2, target2)) in test_loader:
        data=torch.cat([data1,data2],0)
        target=torch.cat([target1,target2],0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        test_loss += F.nll_loss(output, target,size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
#    test()
    train(epoch)
    test()