# -*- coding: utf-8 -*-
# @Time    : 2019/8/6 17:15
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Classification_Cats_Dogs.py
# @Software: PyCharm

import os
import time
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim
import torch

data_path = "E:/pytorch_data/dogs-vs-cats/"


def data_procress():
    files = glob(os.path.join(data_path, '*.jpg'))
    image_nums = len(files)
    print(f'图片总数:{image_nums}')
    shuffle = np.random.permutation(image_nums)
    os.mkdir(os.path.join(data_path, 'valid'))
    # 使用标签名称创建目录
    for t in ['train/', 'valid/']:
        for folder in ['dog', 'cat']:
            os.makedirs(os.path.join(data_path, t, folder))

    for i in shuffle[:2000]:
        folder = files[i].split('\\')[-1].split('.')[0]
        image = files[i].split('\\')[-1]
        os.rename(files[i], os.path.join(data_path, 'valid', folder, image))

    for i in shuffle[2000:]:
        folder = files[i].split('\\')[-1].split('.')[0]
        image = files[i].split('\\')[-1]
        os.rename(files[i], os.path.join(data_path, 'train', folder, image))


def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*30)
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                # 此时为训练模式
                model.train(True)
            else:
                # 此时为评估模式
                model.train(False)
            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, lables = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    lables = Variable(lables.cuda())
                else:
                    inputs, lables = Variable(inputs), Variable(lables)
                # 梯度清零
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, lables)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == lables.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'vaod' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 加载最有权重
    model.load_state_dict(best_model_wts)
    return model

data_procress()
# if __name__ == '__main__':
#
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     train = ImageFolder(data_path + 'train/', simple_transform)
#     valid = ImageFolder(data_path + 'valid/', simple_transform)
#     print('>>>>>>>>>>>>>>>>>>>>>>>.')
#     print(train.class_to_idx)
#     print(train.classes)
#     # imshow(train[500][0])
#     # plt.show()
#     train_data_gen = DataLoader(train, batch_size=64, num_workers=1)
#     valid_data_gen = DataLoader(valid, batch_size=64, num_workers=1)
#     dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
#     dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}
#     model_ft = models.resnet18(pretrained=True)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, 2)
#     if torch.cuda.is_available():
#         print("使用GPU加速")
#         model_ft = model_ft.cuda()
#     learning_rate = 0.001
#     criterion = nn.CrossEntropyLoss()
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#     exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#     torch.cuda.empty_cache()
#     train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    # try:
    #     pass
    # except RuntimeError as e:
    #     if 'out of memory' in str(e):
    #         print('| WARNING: ran out of memory')
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #     else:
    #         raise e














