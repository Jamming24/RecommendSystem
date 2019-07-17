# -*- coding: utf-8 -*-
# @Time    : 2019/6/20 19:14
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Multi_Classifier_CIQ.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from Fire2019_CIQ.Binary_Classifier_CIQ import getTF_IDF_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from Fire2019_CIQ import data_process


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


def load_wordVector(text, wordvector_path):
    # 取词向量加和求平均 作为特征
    vectors = []
    word_vectors = {}
    dimension = 0
    data_csv = pd.read_csv(open(text, encoding='UTF-8'), sep='\t')
    question_text = data_csv["question_text"]
    all_wordsVector = open(wordvector_path, encoding='UTF-8')
    flag = True
    for line in all_wordsVector:
        t = line.strip('\n').split('\t')
        if flag:
            dimension = int(t[1])
            print(f"总次数:{t[0]}, 向量维数:{t[1]}")
            flag = False
            continue
        else:
            word = t[0]
            word_vector_str = t[1].split(',')
            temp = []
            for str in word_vector_str:
                temp.append(float(str))
            word_vectors[word] = np.array(temp)
    for text in question_text:
        count = 0
        text_vector = np.zeros(dimension)
        words = text.strip('\n').split(' ')
        for w in words:
            if w in word_vectors.keys():
                count += 1
                text_vector += word_vectors[w]
        if count > 0:
            vectors.append(text_vector/count)
    return vectors


def load_word_vector_features(CIQ_train, CIQ_dev):
    vector_file = root_path + "实验文件/CIQ_wordVector.tsv"
    train_vector = load_wordVector(text=CIQ_train, wordvector_path=vector_file)
    dev_vector = load_wordVector(text=CIQ_dev, wordvector_path=vector_file)
    train_csv = pd.read_csv(open(CIQ_train, encoding="UTF-8"), sep="\t")
    dev_csv = pd.read_csv(open(CIQ_dev, encoding="UTF-8"), sep="\t")
    train_label = train_csv["target"]
    dev_label = dev_csv["target"]
    return train_vector, train_label, dev_vector, dev_label


def load_tf_idf_features(CIQ_train, CIQ_dev):
    train_csv = pd.read_csv(open(CIQ_train, encoding="UTF-8"), sep="\t")
    dev_csv = pd.read_csv(open(CIQ_dev, encoding="UTF-8"), sep="\t")
    train_data = train_csv["question_text"].tolist()
    # train_data = data_process.text_stemmer(train_csv["question_text"], flag=1).tolist()
    train_label = train_csv["target"]
    dev_data = dev_csv["question_text"].tolist()
    # dev_data = data_process.text_stemmer(dev_csv["question_text"], flag=1).tolist()
    dev_label = dev_csv["target"]

    tf_idf_vertorize = getTF_IDF_matrix(train_data)
    train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
    dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
    return train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label


def multi_LR_classifier(train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label):
    # sgd_clf = SGDClassifier(random_state=42)
    # sgd_clf.fit(train_tf_idf_vectorize, train_label)
    # print(sgd_clf.decision_function(dev_tf_idf_vectorize))
    # y_predict = sgd_clf.predict(dev_tf_idf_vectorize)
    # 多元逻辑回归
    # print("多元逻辑回归")
    # 参数：multionmial,solver="lbfgs", C=10
    softmax_reg = LogisticRegression(max_iter=10)
    softmax_reg.fit(train_tf_idf_vectorize, train_label)
    y_predict = softmax_reg.predict(dev_tf_idf_vectorize)
    # predict() 输出预测类别， predict_proba()输出预测概率
    # decision_function 返回实例分数
    classifier_report = classification_report(dev_label, y_predict)
    print("逻辑回归预测准确率", accuracy_score(dev_label, y_predict), sep=":")
    # print(classifier_report)


def multi_svm_classifier(train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label):
    # print("SVM分类器")
    # 这个好用
    poly_kernel_svm = SVC(decision_function_shape='ovo', kernel="linear", C=1)
    # poly_kernel_svm = LinearSVC(C=1, loss="hinge")
    # poly_kernel_svm = SVC(kernel="poly", degree=2, coef0=10, C=1)

    poly_kernel_svm.fit(train_tf_idf_vectorize, train_label)
    y_predict = poly_kernel_svm.predict(dev_tf_idf_vectorize)
    print("SVM准确率", accuracy_score(y_true=dev_label, y_pred=y_predict), sep=":")


def multi_naive_bayes(train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label):
    # print("朴素贝叶斯分类器")
    NB_clf = MultinomialNB(alpha=0.01)
    NB_clf.fit(train_tf_idf_vectorize, train_label)
    y_preds = NB_clf.predict(dev_tf_idf_vectorize)
    print("朴素贝叶斯准确率", accuracy_score(y_true=dev_label, y_pred=y_preds), sep=':')


def multi_KNN(train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label, iter):
    # print("KNN分类器")
    for x in range(1, iter):
        knnclf = KNeighborsClassifier(n_neighbors=x)
        knnclf.fit(train_tf_idf_vectorize, train_label)
        preds = knnclf.predict(dev_tf_idf_vectorize)
    print("knn准确率", accuracy_score(y_true=dev_label, y_pred=preds), sep=":")


def multiDecisionTree(train_tf_idf_vectorize, train_label, dev_tf_idf_vectorize, dev_label):
    tree_clf = DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(train_tf_idf_vectorize, train_label)
    y_preds = tree_clf.predict(dev_tf_idf_vectorize)
    print("决策树准确率", accuracy_score(y_true=dev_label, y_pred=y_preds), sep=':')


def mian():
    root_path = "E:/Fire2019/评测任务三/"
    model_path = root_path + "模型保存/SVM_default_large.pkl"
    min_train = root_path + "实验文件/train.tsv"
    CIQ_test_data_path = root_path + "实验文件/CIQ_dev.tsv"
    get_CIQ_predict_one(min_train, CIQ_test_data_path, model_path)


root_path = "E:/Fire2019/评测任务三/"
CIQ_train_path = root_path + "实验文件/CIQ_train.tsv"
CIQ_dev_path = root_path + "实验文件/CIQ_dev.tsv"

train_tf_idf, train_label, dev_tf_idf, dev_label = load_tf_idf_features(CIQ_train_path, CIQ_dev_path)
# train_vector, train_label, dev_vector, dev_label = load_word_vector_features(CIQ_train_path, CIQ_dev_path)
# 对于逻辑回归正好想反，用C控制其正则化，C时alpha的逆反，C越大 正则化程度越高
multi_LR_classifier(train_tf_idf, train_label, dev_tf_idf, dev_label)
multi_svm_classifier(train_tf_idf, train_label, dev_tf_idf, dev_label)
multi_KNN(train_tf_idf, train_label, dev_tf_idf, dev_label, iter=10)
multi_naive_bayes(train_tf_idf, train_label, dev_tf_idf, dev_label)
multiDecisionTree(train_tf_idf, train_label, dev_tf_idf, dev_label)

# y_true = [1, 1, 1, 2, 2, 1]
# y_pre = [1, 1, 1, 2, 2, 2]
# print("class_report", classification_report(y_true=y_true, y_pred=y_pre), sep=':')
# print("accuracy_score", accuracy_score(y_true=y_true, y_pred=y_pre), sep=':')
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

# 1    488 0.54
# 3    216 0.24
# 2     98 0.11
# 5     38 0.04
# 4     38 0.04
# 0     20 0.02
