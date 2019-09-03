# -*- coding: utf-8 -*-
# @Time    : 2019/8/19 15:28
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : CNN_Visualizing.py
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


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class LayerActivations():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def remove(self):
        self.hook.remove()


if __name__ == '__main__':
    simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train = ImageFolder(data_path + 'train/', simple_transform)
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=False)
    img, label = next(iter(train_data_loader))
    # imshow(img[5])
    # plt.show()
    img = img[5][None]
    os.environ['TORCH_HOME'] = 'E:/pytorch_data/models/'
    vgg = models.vgg16(pretrained=True).cuda()
    conv_out = LayerActivations(vgg.features, 0)
    o = vgg(Variable(img.cuda()))
    conv_out.remove()
    act = conv_out.features
    fig = plt.figure(figsize=(20, 50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][i])

    conv_out = LayerActivations(vgg.features, 1)
    o = vgg(Variable(img.cuda()))
    conv_out.remove()
    act = conv_out.features
    fig = plt.figure(figsize=(20, 50))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][i])

    # CNN层的可视化权重
    vgg.state_dict().keys()
    cnn_weights = vgg.state_dict()['features.0.weight'].cpu()
    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    for i in range(30):
        ax = fig.add_subplot(12, 6, i + 1, xticks=[], yticks=[])
        imshow(cnn_weights[i])
    print(cnn_weights.shape)
    plt.show()