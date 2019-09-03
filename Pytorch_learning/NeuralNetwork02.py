# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 12:55
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : NeuralNetwork02.py
# @Software: PyCharm

from glob import glob
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch


class DogsAndCatsDataset(Dataset):

    def __init__(self, root_dir, size=(224, 224)):
        # 进行必要的初始化
        self.files = glob(root_dir)
        self.size = size

    def __len__(self):
        # 负责返回数据集中最大元素个数
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img, label


# dataloader = DataLoader(DogsAndCatsDataset, batch_size=32, num_workers=2)
# for img, label in dataloader:
#     pass


# inp = Variable(torch.randn(1, 10))
# 输入为10维张量 输出为5维张量
# myLayer = Linear(in_features=10, out_features=5, bias=True)
# print(myLayer(inp))
# print(myLayer.weight)
# print(myLayer.bias)
# 将一层输出传递给另一层
# myLayer1 = Linear(in_features=10, out_features=5)
# myLayer2 = Linear(in_features=5, out_features=2)
# a = myLayer2(myLayer1(inp))
# print(a)

# sample_data = Variable(torch.Tensor([[1, 2, -1, -1]]))
# 把sample_data直接放进ReLU()会报错，这是为何????
# myRelu = ReLU()
# print(myRelu(sample_data))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()
        num_feature = 1
        for s in size:
            num_feature *= s
        return num_feature


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
input = torch.randn(1, 1, 32, 32, requires_grad=True)
out = net(input)
print(out)
out.backward(torch.randn(1, 10))
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
criterion = nn.MSELoss()
loss = criterion(out, target)
# loss.backward(retain_graph=True)
optimizer.step()
print(loss)
