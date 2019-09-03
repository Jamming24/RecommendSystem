# -*- coding: utf-8 -*-
# @Time    : 2019/8/20 11:20
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : ResNet_Pytorch.py
# @Software: PyCharm
import pandas as pd
from glob import glob
import os
from shutil import copyfile

from torch import optim
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18,resnet34
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

import pickle

is_cuda = torch.cuda.is_available()


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class FullyConnectedModel(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, inp):
        out = self.fc(inp)
        return out


class FeaturesDataset(Dataset):

    def __init__(self, featlst, labellst):
        self.featlst = featlst
        self.labellst = labellst

    def __getitem__(self, index):
        return (self.featlst[index], self.labellst[index])

    def __len__(self):
        return len(self.labellst)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).data[0]
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


if __name__ == '__main__':

    data_transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_path = "E:/pytorch_data/dogs-vs-cats/"
    train_dset = ImageFolder(data_path + 'train/', transform=data_transform)
    val_dset = ImageFolder(data_path + 'valid/', transform=data_transform)
    classes=2
    train_loader = DataLoader(train_dset,batch_size=32,shuffle=False,num_workers=3)
    val_loader = DataLoader(val_dset,batch_size=32,shuffle=False,num_workers=3)
    my_resnet = resnet34(pretrained=True)

    if is_cuda:
        my_resnet = my_resnet.cuda()

    m = nn.Sequential(*list(my_resnet.children())[:-1])
    trn_labels = []

    # Stores the pre convoluted features of the train data
    trn_features = []

    #Iterate through the train data and store the calculated features and the labels
    for d,la in train_loader:
        o = m(Variable(d.cuda()))
        o = o.view(o.size(0),-1)
        trn_labels.extend(la)
        trn_features.extend(o.cpu().data)

    #For validation data

    #Iterate through the validation data and store the calculated features and the labels
    val_labels = []
    val_features = []
    for d,la in val_loader:
        o = m(Variable(d.cuda()))
        o = o.view(o.size(0),-1)
        val_labels.extend(la)
        val_features.extend(o.cpu().data)
    trn_feat_dset = FeaturesDataset(trn_features,trn_labels)
    val_feat_dset = FeaturesDataset(val_features,val_labels)

    #Creating data loader for train and validation
    trn_feat_loader = DataLoader(trn_feat_dset,batch_size=64,shuffle=True)
    val_feat_loader = DataLoader(val_feat_dset,batch_size=64)

    fc_in_size = 8192
    fc = FullyConnectedModel(fc_in_size,classes)
    if is_cuda:
        fc = fc.cuda()
    optimizer = optim.Adam(fc.parameters(),lr=0.0001)
    train_losses , train_accuracy = [],[]
    val_losses , val_accuracy = [],[]
    for epoch in range(1,10):
        epoch_loss, epoch_accuracy = fit(epoch,fc,trn_feat_loader,phase='training')
        val_epoch_loss , val_epoch_accuracy = fit(epoch,fc,val_feat_loader,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
