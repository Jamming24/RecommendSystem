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
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
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


def test_OneVSOne():
    mnist = fetch_mldata("MNIST original", data_home='./datasets')
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = X[:6000], X[6000:6500], y[:6000], y[6000:6500]
    shuffle_index = np.random.permutation(6000)
    # 对60000个数进行洗牌
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    some_digit = X[360]
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    print(svm_clf.predict([some_digit]))
    some_digit_scores = svm_clf.decision_function([some_digit])
    print(some_digit_scores)
    # classes_可以查询分类类别
    print(svm_clf.classes_)
    # 强制使用一对一策略
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(X_train, y_train)
    print(ovo_clf.predict([some_digit]))
    print(len(ovo_clf.estimators_))


test_OneVSOne()
