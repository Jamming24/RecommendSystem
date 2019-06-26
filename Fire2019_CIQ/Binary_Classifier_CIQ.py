# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 0:34
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Binary_Classifier_CIQ.py
# @Software: PyCharm


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


def CIQ_LR(train_X, train_y, test_X, test_y, save_path):
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(train_X, train_y)
    # 打印参数
    # print(log_reg.coef_)
    predict_y = log_reg.predict(test_X)
    print(classification_report(test_y, predict_y))
    # 保存模型
    joblib.dump(log_reg, save_path)
    # 打印混淆矩阵
    print(confusion_matrix(test_y, predict_y))


def CIQ_SVM(train_X, train_y, test_X, test_y, save_path):
    # svm_clf = Pipeline(("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge")))
    svm_clf = LinearSVC(C=1, loss="hinge", class_weight='balanced')
    svm_clf.fit(train_X, train_y)
    predict_y = svm_clf.predict(test_X)
    print(classification_report(test_y, predict_y))
    # 保存模型
    joblib.dump(svm_clf, save_path)
    # 打印混淆矩阵
    print(confusion_matrix(test_y, predict_y))


def getTF_IDF_matrix(question_text):
    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True)
    vectorizer.fit(question_text)
    # word_list = vectorizer.get_feature_names()
    # print(len(word_list))
    # print(tf_idf_vectorizer.toarray())
    return vectorizer


def load_data(tsv_path):
    train_tsv = pd.read_csv(open(tsv_path, encoding='UTF-8'), sep='\t', index_col=False)
    # print(train_tsv.tail(5))
    question_text = train_tsv.loc[:, ['question_text']]
    target = train_tsv.loc[:, ['target']]
    data = np.array(question_text).tolist()
    target = np.array(target).tolist()
    data_all = []
    target_all = []
    for dat in data:
        for simple_data in dat:
            data_all.append(simple_data)
    for tar in target:
        for simple in tar:
            target_all.append(simple)
    return data_all, target_all


def main():
    root_path = "E:/Fire2019/评测任务三/"
    train_tsv_path = root_path + "实验文件/train.tsv"
    test_tsv_path = root_path + "实验文件/dev.tsv"
    LR_save_path = root_path+"模型保存/LR_default_large.pkl"
    SVM_save_path = root_path+"模型保存/SVM_default_large.pkl"
    train_data, train_target = load_data(train_tsv_path)
    test_data, test_traget = load_data(test_tsv_path)

    tf_idf_vectorizer = getTF_IDF_matrix(train_data)
    train_tf_idf_vectorizer = tf_idf_vectorizer.transform(train_data)
    test_tf_idf_vectorizer = tf_idf_vectorizer.transform(test_data)

    print("LR分类器结果")
    CIQ_LR(train_tf_idf_vectorizer, train_target, test_tf_idf_vectorizer, test_traget, save_path=LR_save_path)
    print("SVM分类器结果")
    CIQ_SVM(train_tf_idf_vectorizer, train_target, test_tf_idf_vectorizer, test_traget, save_path=SVM_save_path)


