# -*- coding: utf-8 -*-
# @Time    : 2019/8/19 13:09
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : CNN_Cats_Dogs.py
# @Software: PyCharm

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

data_path = "E:/pytorch_data/dogs-vs-cats/"
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # view操作时将二维张量转换为一维向量，以便于线性层处理
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def base_CNN_fit(epoch, model, data_loader, phase='training', volatile=False):
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
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = running_correct.__float__() / len(data_loader.dataset)
    print(
        f'第{epoch}次迭代, {phase} loss is {loss} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)},{accuracy}')
    return loss, accuracy


def Vgg_fit(epoch, model, data_loader, phase='training', volatile=False):
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
    accuracy = running_correct.__float__() / len(data_loader.dataset)
    print(
        f'{phase} loss is {loss.__float__()} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy}')
    return loss, accuracy


def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return conv_features, labels_list


class My_dataset(Dataset):
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]


def data_gen(conv_feat,labels,batch_size=64,shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0,len(conv_feat),batch_size):
        yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])


def fit_numpy(epoch, model, data_loader, phase='training', volatile=False):
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
        data = data.view(data.size(0), -1)
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).data[0]
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = running_correct.__float__() / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss.__float__()} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy}')
    return loss, accuracy


if __name__ == '__main__':
    # 使用动态翻转等操作 增强模型泛化能力
    simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(0.2), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train = ImageFolder(data_path + 'train/', simple_transform)
    valid = ImageFolder(data_path + 'valid/', simple_transform)
    print(train.class_to_idx)
    print(train.classes)
    # imshow(valid[770][0])
    # plt.show()
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=32, num_workers=3, shuffle=True)
    # model = Net()
    # if is_cuda:
    #     model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #
    # train_losses, train_accuracy = [], []
    # val_losses, val_accuracy = [], []
    # for epoch in range(1, 4):
    #     epoch_loss, epoch_accuracy = base_CNN_fit(epoch, model, train_data_loader, phase='training')
    #     val_epoch_loss, val_epoch_accuracy = base_CNN_fit(epoch, model, valid_data_loader, phase='validation')
    #     train_losses.append(epoch_loss)
    #     train_accuracy.append(epoch_accuracy)
    #     val_losses.append(val_epoch_loss)
    #     val_accuracy.append(val_epoch_accuracy)

    # 显存太小了 2GB显存根本train不动VGG
    os.environ['TORCH_HOME'] = 'E:/pytorch_data/models/'
    vgg = models.vgg16(pretrained=True)
    vgg = vgg.cuda()

    # print(vgg)
    # 冻结features层的所有特征权重，防止训练模型的时候更新这些权重
    for param in vgg.features.parameters():
        param.requires_grad = False
    features = vgg.features
    # 计算预卷积特征之后 进行训练
    conv_feat_train, labels_train = preconvfeat(train_data_loader, features)
    conv_feat_val, labels_val = preconvfeat(valid_data_loader, features)

    train_feat_dataset = My_dataset(conv_feat_train, labels_train)
    val_feat_dataset = My_dataset(conv_feat_val, labels_val)

    train_feat_loader = DataLoader(train_feat_dataset, batch_size=64, shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset, batch_size=64, shuffle=True)

    train_batches = data_gen(conv_feat_train, labels_train)
    val_batches = data_gen(conv_feat_val, labels_val)
    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit_numpy(epoch, vgg.classifier, val_feat_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)


    # 冻结features层之间训练
    # vgg.classifier[6].out_features = 2
    # optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
    # # 使用dropout进行优化权重
    # for layer in vgg.classifier.children():
    #     if type(layer) == nn.Dropout:
    #         layer.p = 0.2
    # train_losses, train_accuracy = [], []
    # val_losses, val_accuracy = [], []
    # for epoch in range(1, 10):
    #     epoch_loss, epoch_accuracy = Vgg_fit(epoch, vgg, train_data_loader, phase='training')
    #     val_epoch_loss, val_epoch_accuracy = Vgg_fit(epoch, vgg, valid_data_loader, phase='validation')
    #     train_losses.append(epoch_loss)
    #     train_accuracy.append(epoch_accuracy)
    #     val_losses.append(val_epoch_loss)
    #     val_accuracy.append(val_epoch_accuracy)

    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
    plt.legend()
    plt.show()
