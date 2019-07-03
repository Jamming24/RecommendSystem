# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 19:14
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Multi_Classifier_CIQ.py
# @Software: PyCharm

import pandas as pd
from sklearn.externals import joblib
from Fire2019_CIQ.Binary_Classifier_CIQ import getTF_IDF_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from Fire2019_CIQ import data_process


# 如果二分类能够将所有的虚假问题都能跳出来，即 1标签的召回率为100% 进行五分类，不足100% 进行二分类

# han:取出标签是1 的做六分类， 或者5分类, 二分类用机器学习做， 多分类用bert做


def get_CIQ_predict_one(min_train_path, CIQ_test_data, save_path):
    log_reg = joblib.load(save_path)

    # print(log_reg.coef_)
    mini_train_data = pd.read_csv(open(min_train_path, encoding="UTF-8"), sep="\t")
    CIQ_test_data = pd.read_csv(open(CIQ_test_data, encoding="UTF-8"), sep="\t")
    train_texts = mini_train_data["question_text"].tolist()
    test_texts = CIQ_test_data["question_text"].tolist()
    test_labels = CIQ_test_data["target"].tolist()
    binary_test_labels = []
    for lable in test_labels:
        if lable == 0:
            binary_test_labels.append(0)
        else:
            binary_test_labels.append(1)
    tf_idf_vectorize = getTF_IDF_matrix(train_texts)
    # 将全部数据作为test进行预测零一
    test_tf_idf_vectorize = tf_idf_vectorize.transform(test_texts)
    y_predict_CIQ = log_reg.predict(test_tf_idf_vectorize)
    predict_one = []
    for i in range(len(y_predict_CIQ)):
        if y_predict_CIQ[i] == 1:
            predict_one.append(i)
    # 用SVM_large版本模型在新的900条数据集上进行二分类
    print("SVM分类结果")
    print(classification_report(binary_test_labels, y_predict_CIQ))
    predict_one_csv = CIQ_test_data.loc[predict_one]
    predict_one_csv.to_csv(root_path + "实验文件/CIQ_dev_predict_one.tsv", encoding='UTF-8', sep='\t', index=False)
    print("开发集合中用SVM二分类预测为一的文件输出完成")


def multi_LR_classifier(CIQ_train, CIQ_dev):
    train_csv = pd.read_csv(open(CIQ_train, encoding="UTF-8"), sep="\t")
    dev_csv = pd.read_csv(open(CIQ_dev, encoding="UTF-8"), sep="\t")
    # train_data = train_csv["question_text"].tolist()
    train_data = data_process.text_stemmer(train_csv["question_text"], flag=1).tolist()
    train_label = train_csv["target"]
    # dev_data = dev_csv["question_text"].tolist()
    dev_data = data_process.text_stemmer(dev_csv["question_text"], flag=1).tolist()
    dev_label = dev_csv["target"]

    tf_idf_vertorize = getTF_IDF_matrix(train_data)
    train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
    dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
    # 训练模型
    softmax_reg = LogisticRegression(multi_class="multionmial", solver="lbfgs", C=10)
    softmax_reg.fit(train_tf_idf_vectorize, train_label)
    # 得到模型
    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(train_tf_idf_vectorize, train_label)
    # 多元逻辑回归
    print("多元逻辑回归")
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=15)
    softmax_reg.fit(train_tf_idf_vectorize, train_label)
    y_predict = softmax_reg.predict(dev_tf_idf_vectorize)
    # predict() 输出预测类别， predict_proba()输出预测概率
    # decision_function 返回实例分数
    # print(sgd_clf.decision_function(dev_tf_idf_vectorize))
    # y_predict = sgd_clf.predict(dev_tf_idf_vectorize)
    classifier_report = classification_report(dev_label, y_predict)
    print(classifier_report)


def multi_svm_classifier(CIQ_train, CIQ_dev):
    train_csv = pd.read_csv(open(CIQ_train, encoding="UTF-8"), sep="\t")
    dev_csv = pd.read_csv(open(CIQ_dev, encoding="UTF-8"), sep="\t")
    train_data = train_csv["question_text"].tolist()
    train_label = train_csv["target"]
    dev_data = dev_csv["question_text"].tolist()
    dev_label = dev_csv["target"]

    tf_idf_vertorize = getTF_IDF_matrix(train_data)
    train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
    dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
    print("多项式SVM")

    poly_kernel_svm = SVC(decision_function_shape='ovo')
    poly_kernel_svm.fit(train_tf_idf_vectorize, train_label)
    y_predict = poly_kernel_svm.predict(dev_tf_idf_vectorize)
    # predict() 输出预测类别， predict_proba()输出预测概率
    # decision_function 返回实例分数
    # print(sgd_clf.decision_function(dev_tf_idf_vectorize))
    # y_predict = sgd_clf.predict(dev_tf_idf_vectorize)
    classifier_report = classification_report(dev_label, y_predict)
    print(classifier_report)


def mian():
    root_path = "E:/Fire2019/评测任务三/"
    model_path = root_path + "模型保存/SVM_default_large.pkl"
    min_train = root_path + "实验文件/train.tsv"
    CIQ_test_data_path = root_path + "实验文件/CIQ_dev.tsv"
    get_CIQ_predict_one(min_train, CIQ_test_data_path, model_path)


root_path = "E:/Fire2019/评测任务三/"
CIQ_train_path = root_path + "实验文件/CIQ_train.tsv"
CIQ_dev_path = root_path + "实验文件/CIQ_dev.tsv"
multi_LR_classifier(CIQ_train_path, CIQ_dev_path)
multi_svm_classifier(CIQ_train_path, CIQ_dev_path)
# -----------------------
# 对数据进行去停用词 取词干 词型还原等操作
# 加入embedding作为特征 word2vector doc2vector
# 分类器加入 随机森林
# -------------------------
# csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep='\t')
# csv.hist(bins=5, figsize=(10, 5))
# plt.show()
# value_counts可以计算出各个类别的数量
# print(all_data["target"].value_counts())

# 1    488
# 3    216
# 2     98
# 5     38
# 4     38
# 0     20
