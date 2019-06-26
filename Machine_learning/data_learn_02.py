# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 14:25
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : datadata_learn_02.py
# @Software: PyCharm

import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
# 返回分层划分，可以保证数据中每一个划分中类的样本比例与整体数据原始比例保持一致 shuffle用于打乱顺序
from sklearn.model_selection import StratifiedKFold
# 保证数据中每一类的比例都是相同的

# six是用来兼容python 2 和 3的
# six.moves 是用来处理那些在2 和 3里面函数的位置有变化的，直接用six.moves就可以屏蔽掉这些变化

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housing_path = "datasets/housing"
housing_url = DOWNLOAD_ROOT + housing_path + "/housing.tgz"


def fetch_housing_data(housing_url, housing_path):
    # 从github上下载数据集
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    # 开始下载
    print("开始下载")
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print("下载完成")
# fetch_housing_data(housing_url, housing_path)


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    # 产生一个随机序列，用来打散训练数据
    # 随机数种子生成器 可以保证每次生成的随机序列都一样  42是随机取的 来自银河系漫游指南
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    # print(shuffled_indices)
    test_set_size = int(len(data) * test_ratio)
    # 测试数据的随机序列索引
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloca 整形索引  用来取出指定列
    return data.iloc[train_indices], data.iloc[test_indices]


# 设置pandas显示所有隐藏行和列
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
housing = load_housing_data(housing_path)
# 将收入中位数处以1.5 然后用ceil取整 作为新类别
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
# 将大于5的类别合并为类别5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# 进行分层切分数据集 可以维持训练数据和测试集合中的数据均衡
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

# print(housing["income_cat"].value_counts()/len(housing))

# 删除某一列 axis=1 表示 是删除的是列 inplace=True 改变原数据
for set in (start_test_set, start_train_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")
# 直接用sk工具包进行切分数据集合
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# 数据分析
# pandas loc通过行和列的名称进行索引访问 iloc通过整数 或者整数数列进行索引 ix 是iloc和loc的整合
# print(housing.head())
# 可以快速获取数据集信息
# print(housing.info())
# 按照值分组
# print(housing["ocean_proximity"].value_counts())
# 将列值取出来  并转换成list输出
# print(housing["ocean_proximity"].tolist())
# 打印属性摘要 其中std表示的是标准差  比如housing_median_age的 25% 表示 25%的数值低于18
# print(housing.describe())
# 用pandas绘制数据直方图 bins 指的是箱子的个数，即直方图的柱子数 figsize 指的是每张图的大小
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


