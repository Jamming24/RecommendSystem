# -*- coding: utf-8 -*-
# @Time    : 2019/9/3 8:27
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : LR_Enhance.py
# @Software: PyCharm

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

root_path = "E:/Fire2019/评测任务三/"


def getTF_IDF_matrix(question_text):
    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True)
    vectorizer.fit(question_text)
    # word_list = vectorizer.get_feature_names()
    # print(len(word_list))
    # print(tf_idf_vectorizer.toarray())
    return vectorizer


def get_features():
    CIQ_train_path = root_path + "实验文件/CIQ_all_train.tsv"
    CIQ_dev_path = root_path + "实验文件/CIQ_test_no_Label.tsv"

    train_csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep="\t")
    dev_csv = pd.read_csv(open(CIQ_dev_path, encoding="UTF-8"), sep="\t")

    train_data = train_csv["question_text"].tolist()
    train_label = train_csv["target"]
    dev_data = dev_csv["question_text"].tolist()
    dev_label = dev_csv["target"]

    tf_idf_vertorize = getTF_IDF_matrix(train_data)
    train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
    dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
    return train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label


def binary_data_process(y, save_positive=1):
    print("转换前的标签分布")
    print(y.value_counts())
    for index in y.index:
        if y.loc[index] != save_positive:
            y.loc[index] = 0
    print("转换后的标签分布")
    print(y.value_counts())
    return y


def binary_LogisticRegression(binary_train_x, binary_train_y, threshold):
    point_index = 0
    margin_point = []
    binary_lr = LogisticRegression(solver="lbfgs")
    binary_lr.fit(binary_train_x, binary_train_y)
    probability = binary_lr.predict_proba(binary_train_x)
    for prob in probability:
        if abs(prob[1] -prob[0]) < threshold:
            margin_point.append(point_index)
        point_index += 1
    return margin_point


def LogisticRegression_Enhance(X_data, y_label, threshold, max_iter, dev_data, dev_label):
    binary_train_y = binary_data_process(y_label, 1)
    binary_dev_y = binary_data_process(dev_label, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X_data, binary_train_y)
    predict = lr.predict(dev_data)
    print(accuracy_score(y_pred=predict, y_true=binary_dev_y))

    margin_point_index = binary_LogisticRegression(X_data, binary_train_y, threshold)
    margin_data = X_data[margin_point_index]
    margin_label = y_label[margin_point_index].reset_index(drop=True)
    lr_enhance = None
    for iter in range(max_iter):
        print(f"第{iter}迭代， 分类边界数据量{len(margin_point_index)}")
        lr_enhance = LogisticRegression(solver="lbfgs")
        lr_enhance.fit(margin_data, margin_label)
        predict = lr_enhance.predict(dev_data)
        print(accuracy_score(y_pred=predict, y_true=binary_dev_y))
        margin_point_index = binary_LogisticRegression(margin_data, margin_label, threshold)
        margin_data = margin_data[margin_point_index]
        margin_label = margin_label[margin_point_index].reset_index(drop=True)
    return lr_enhance


def LR_Enhance_multi_classifier(train_X, train_y, test_X, test_y):
    # 训练数据有几个类别就训练几个二分类模型
    a = train_y.value_counts()
    for classifier in a.index:
        print(classifier)
        binary_train_y = binary_data_process(test_X, 1)
        binary_dev_y = binary_data_process(dev_label, 1)


train_TF_IDF, train_label, dev_TF_IDF, dev_label = get_features()
LR_Enhance_multi_classifier(train_TF_IDF, train_label, dev_TF_IDF, dev_label)
# LogisticRegression_Enhance(train_TF_IDF, train_label, 0.2, 3, dev_TF_IDF, dev_label)

# binary_dev_y = binary_data_process(dev_label, 1)
