# -*- coding: utf-8 -*-
# @Time    : 2019/6/19 22:25
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : abandon_code.py
# @Software: PyCharm

import pandas as pd
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


root_path = "E:/Fire2019/评测任务三/"


def generating_data():
    # 总数：1306118，1225308个0 占比0.93812；80810个1 占比 0.06187  0是正常的 1 是虚假问题
    csv_data = pd.read_csv(open(root_path + '源文件/train.csv', encoding='UTF-8'))
    print(csv_data.shape)
    # 读取后10行
    # train_data = csv_data.tail(10)
    # print(train_data)
    # 切分数据
    train_data = csv_data.loc[0:914283]
    dev_data = csv_data.loc[914284:1175507]
    test_data = csv_data.loc[1175508:1306118]
    print("生成train_data")
    train_data.to_csv(root_path + "实验文件/train.tsv", sep='\t', encoding='UTF-8', index=False)
    print("生成dev_data")
    dev_data.to_csv(root_path + "实验文件/dev.tsv", sep='\t', encoding='UTF-8', index=False)
    print("生成test_data")
    test_data.to_csv(root_path + "实验文件/test.tsv", sep='\t', encoding='UTF-8', index=False)
    # print("最后一行")
    # print(csv_data.tail(2))


def generating_mini_data():
    csv_data = pd.read_csv(open(root_path + '源文件/train.csv', encoding='UTF-8'))
    print(csv_data.shape)
    train_data = csv_data.loc[0:6999]
    dev_data = csv_data.loc[7000:8999]
    test_data = csv_data.loc[9000:9999]
    print("生成train_mini_data")
    train_data.to_csv(root_path + "实验文件/train_mini.tsv", sep='\t', encoding='UTF-8', index=False)
    print("生成dev_mini_data")
    dev_data.to_csv(root_path + "实验文件/dev_mini.tsv", sep='\t', encoding='UTF-8', index=False)
    print("生成test_mini_data")
    test_data.to_csv(root_path + "实验文件/test_mini.tsv", sep='\t', encoding='UTF-8', index=False)

def test():
    iris = datasets.load_iris()
    print(list(iris.keys()))
    X = iris["data"][:, 3:]
    y = (iris["target"] == 2).astype(np.int)

    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    X_new = np.linspace(0, 3, 100).reshape(-1, 1)
    y_proda = log_reg.predict_proba(X_new)
    plt.plot(X_new, y_proda[:, 1], "g-", label="Iris-Virginica")
    plt.plot(X_new, y_proda[:, 0], "b--", label="Not Iris-Virginica")
    print(log_reg.predict([[1.7], [1.5]]))
    plt.show()


def get_tf_idf():
    corpus = ['This is the first document.', 'This is the second second document.', 'And the third one.',
              'Is this the first document?']

    vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
    tf_idf_vectorizer = vectorizer.fit_transform(corpus)
    word_list = vectorizer.get_feature_names()
    print(tf_idf_vectorizer.toarray())

    # vectorizer = TfidfVectorizer(min_df=1)
    # vectorizer.fit_transform(corpus)
    # word_list = vectorizer.get_feature_names()
    # tf_idf_vectorizer = TfidfVectorizer(min_df=1).fit_transform(corpus)
    # print(tf_idf_vectorizer.toarray())
    # print(word_list)
    # X = vectorizer.transform(corpus)
    # print(X.toarray())
    # print()
    # print(">>>>>>>>打印词典索引")
    # print(TfidfVectorizer().fit(corpus).vocabulary_)
    # print("<<<<<<<<<")
    # print(TfidfVectorizer().fit(corpus).idf_)
    # print(TfidfVectorizer().fit(corpus).smooth_idf)
    # x = TfidfVectorizer().fit(corpus)
    # print(x.transform(corpus).toarray())
    # print("<<<<<<<<<<<<<<<<")
    # print(vectorizer.fit_transform(corpus).toarray()[1:3])
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # y = [0, 1, 0, 1]
    # # CIQ_LR(X, y)
    # print("训练完成")

