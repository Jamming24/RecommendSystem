# -*- coding: utf-8 -*-
# @Time    : 2019/8/20 11:08
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : InceptionNet_Pytorch.py
# @Software: PyCharm
import pandas as pd
from glob import glob
import os
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class LayerActivations():
    features = []

    def __init__(self, model):
        self.features = []
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.extend(output.view(output.size(0), -1).cpu().data)

    def remove(self):
        self.hook.remove()


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


class FullyConnectedModel(nn.Module):

    def __init__(self, in_size, out_size, training=True):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, inp):
        out = F.dropout(inp, training=self.training)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'E:/pytorch_data/models/'
    data_transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_path = "E:/pytorch_data/dogs-vs-cats/"
    train_dset = ImageFolder(data_path + 'train/', transform=data_transform)
    val_dset = ImageFolder(data_path + 'valid/', transform=data_transform)
    classes=2
    # The size of the output from the selected convolution feature
    fc_in_size = 131072

    fc = FullyConnectedModel(fc_in_size, classes)
    if is_cuda:
        fc = fc.cuda()
    train_loader = DataLoader(train_dset,batch_size=32,shuffle=False,num_workers=3)
    val_loader = DataLoader(val_dset,batch_size=32,shuffle=False,num_workers=3)
    my_inception = inception_v3(pretrained=True)
    my_inception.aux_logits = False
    if is_cuda:
        my_inception = my_inception.cuda()
    trn_features = LayerActivations(my_inception.Mixed_7c)
    trn_labels = []

    # Passing all the data through the model , as a side effect the outputs will get stored
    # in the features list of the LayerActivations object.
    for da, la in train_loader:
        _ = my_inception(Variable(da.cuda()))
        trn_labels.extend(la)
    trn_features.remove()

    # Repeat the same process for validation dataset .

    val_features = LayerActivations(my_inception.Mixed_7c)
    val_labels = []
    for da,la in val_loader:
        _ = my_inception(Variable(da.cuda()))
        val_labels.extend(la)
    val_features.remove()

    trn_feat_dset = FeaturesDataset(trn_features.features,trn_labels)
    val_feat_dset = FeaturesDataset(val_features.features,val_labels)

    #Data loaders for pre computed features for train and validation data sets

    trn_feat_loader = DataLoader(trn_feat_dset,batch_size=64,shuffle=True)
    val_feat_loader = DataLoader(val_feat_dset,batch_size=64)
    optimizer = optim.Adam(fc.parameters(),lr=0.01)
    train_losses , train_accuracy = [],[]
    val_losses , val_accuracy = [],[]
    for epoch in range(1,10):
        epoch_loss, epoch_accuracy = fit(epoch,fc,trn_feat_loader,phase='training')
        val_epoch_loss , val_epoch_accuracy = fit(epoch,fc.eval(),val_feat_loader,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    # optimizer.param_groups[0]['lr']= 0.0001
    #
    # for epoch in range(1,10):
    #     epoch_loss, epoch_accuracy = fit(epoch,fc,trn_feat_loader,phase='training')
    #     val_epoch_loss , val_epoch_accuracy = fit(epoch,fc,val_feat_loader,phase='validation')
    #     train_losses.append(epoch_loss)
    #     train_accuracy.append(epoch_accuracy)
    #     val_losses.append(val_epoch_loss)
    #     val_accuracy.append(val_epoch_accuracy)


