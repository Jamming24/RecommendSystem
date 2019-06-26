# -*- coding: utf-8 -*-
# @Time    : 2019/6/18 23:41
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : data_show_02.py
# @Software: PyCharm
import pandas as pd
# pandas 在0.19以后 把scatter_matrix 移动到 plotting包下了
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from Machine_learning import data_learn_02

# 创建数据副本
start_train_set = data_learn_02.start_train_set
housing = start_train_set.copy()
# 使用drop()用来删除数据，但是删除以后返回一个新的数据DataFrame，而不改变原来的数据 axis=1 表示删除列 默认是行
housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()
# 数据可视化
# housing.plot(kind="scatter", x="longitude", y="latitude")
# alpha 指点的不透明度 当点的透明度很高的时候，单点颜色很浅，这样点越密集，对应区域颜色越深 alpha=0 表示无色
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# print(housing.shape)
# 每个圆的半径大小代表了每个地区的人口数量，我们使用一个名叫jet的预定于颜色表（选项cmap）来进行可视化
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

# 以下用来检验属性之间的相关性
# 标准相关系数 又称皮尔逊相关系数 此方法  可以计算出 每对属性之间的标准相关系数 注意是每对
# corr_matrix = housing.corr()
# 所以  以下是打印出 median_house_value 与其他属性的相关系数
# 相关系数范围从-1到1 越接近1表示越强的正相关 并且相关性与斜率毫无关系
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 使用scatter_matrix绘制出每个属性与其他属性之间的相关性
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# plt.legend()
# plt.show()


