# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 12:39
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : data_process_02.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# 保存模型
joblib.dump(my_model, "my_model.pkl")
my_model_load = joblib.load("my_model.pkl")



# imputer就是一个估算器

housing_path = "datasets/housing/"
csv_path = os.path.join(housing_path, "housing.csv")
housing = pd.read_csv(csv_path)
housing_num = housing.drop("ocean_proximity", axis=1)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]

# 特征缩放
X_train = np.array([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]])
min_max = MinMaxScaler()
print(min_max.fit_transform(X_train))

# 使用流水线处理数据
num_pipeline = Pipeline([('imputer', Imputer(strategy="median")), ('attribs_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler()), ])
num_pipeline.fit_transform(housing_num)

# LabelBinarizer可以一次性完成两个转换（从文本转换为整数类别，在从整数转换为读热编码）
# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat)

# 将文本属性转换成数字编码
housing_cat_encoder = encoder.fit_transform(housing_cat)
print(housing_cat_encoder)
print(encoder.classes_)
# 将文本转换成读热编码
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoder.reshape(-1, 1))
print(housing_cat_1hot.toarray())


enconder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

# 使用Imputer处理数据中的缺失值
# strategy: 'mean'(默认的)， ‘median’中位数，‘most_frequent’出现频率最大的数 axis: 0(默认)， 1 copy: True(默认)，  False

imputer = Imputer(strategy="median")

# 使用fit方法 将imputer实例适配到训练集
imputer.fit(housing_num)
# 说明是按按照中位数进行补足缺失值
# 查看每列的均值
print(imputer.statistics_)
print(housing_num.median().values)
# 转换结果是一个numpy数组
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# 也可以使用fit_transform()完成 相当于先调用fit 在transform
