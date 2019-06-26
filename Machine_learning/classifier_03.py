# -*- coding: utf-8 -*-
# @Time    : 2019/6/21 15:59
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : classifier_03.py
# @Software: PyCharm

from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
# from sklearn.datasets.base import get_data_home
# 获取数据缓存路径
# print(get_data_home())


def data_show(X, y):
    # 70000张照片  784维特征
    print(X.shape)
    print(y.shape)
    some = X[36000]
    # 重塑为28*28的矩阵
    some_digit_image = some.reshape(28, 28)
    # interpolation代表的是插值运算，'nearest'只是选取了其中的一种插值方式。
    # cmap表示绘图时的样式，这里选择的是ocean主题
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    # 去掉横纵坐标轴
    plt.axis("off")
    plt.show()
    print(y[36000])


mnist = fetch_mldata("MNIST original", data_home='./datasets')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
# 对60000个数进行洗牌
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]