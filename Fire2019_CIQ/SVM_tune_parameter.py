# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 19:05
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : SVM_tune_parameter.py
# @Software: PyCharm

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
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


def train_SVM(train_X, train_y, dev_X, dev_y, param_grid):
    # C= 0.01 0.1 0.2 0.5 0.8 0.9 1 5 10
    # 核函数：linear ， poly rbf sigmoid
    # degeree 仅对高斯核有效 rbf
    # gamma 仅仅对rbf，poly sigmoid有效
    # coefo 仅对poly sigmoid有效
    # tol 停止训练的误差精度
    # class_weight=balance 加上之后性能会下降
    # max_iter 最大迭代次数
    # 参数调节 主要设计C，核函数，degree， gamma， coefo， C=0.9 :0.673267
    svm_clf = SVC(probability=True, decision_function_shape='ovo', kernel="linear")
    grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(train_X, train_y)
    print(grid_search.best_params_)
    cvres = grid_search.cv_results_
    # print(cvres)
    for accuracy, para in zip(cvres["mean_test_score"], cvres["params"]):
        print(accuracy, para)


param = [0.195, 0.196, 0.197]
params = [{'C': param}]
train_tf_idf, train_label, dev_tf_idf, dev_label = get_features()
# train_SVM(train_tf_idf, train_label, dev_tf_idf, dev_label, params)
for p in param:
    svm_clf = SVC(probability=True, decision_function_shape='ovo', kernel="linear", C=p)
    svm_clf.fit(train_tf_idf, train_label)
    predict = svm_clf.predict(dev_tf_idf)
    acc = accuracy_score(y_pred=predict, y_true=dev_label)
    print(f"C={p}, 召回率：{acc}")
