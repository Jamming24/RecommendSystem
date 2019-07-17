# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 12:55
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : NeuralNetwork02.py
# @Software: PyCharm

import torch
import numpy as np
# from pip._vendor.packaging.markers import Variable


# 0维度张量
x = torch.rand(10)
print(x)
print(x.size())

def get_data():
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy)