# -*- coding: utf-8 -*-
# @Time    : 2019/6/29 15:10
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : ensemble_learning.py
# @Software: PyCharm
from Fire2019_CIQ.Binary_Classifier_CIQ import getTF_IDF_matrix
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import gensim
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def get_CIQ_wordVector():
    wordlists = PlaintextCorpusReader(root_path + "实验文件", fileids='.*\.tsv')
    words = wordlists.words("CIQ_traindata_contents.tsv")
    model = gensim.models.KeyedVectors.load_word2vec_format(root_path + 'GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    vocabulary = model.wv.vocab.keys()
    file_object = open(root_path + "实验文件/CIQ_wordVector.tsv", encoding='UTF-8', mode='w')
    file_object.write("word\tvector\n")
    for w in words:
        if len(w) != 20 and w in vocabulary:
            vector_list = model.get_vector(w).tolist()
            file_object.write(w + "\t")
            for index in range(len(vector_list) - 1):
                file_object.write(str(vector_list[index]) + ",")
            file_object.write(str(vector_list[len(vector_list) - 1]) + "\n")
    file_object.close()


root_path = "E:/Fire2019/评测任务三/"
CIQ_train_path = root_path + "实验文件/CIQ_train.tsv"
CIQ_dev_path = root_path + "实验文件/CIQ_dev.tsv"
train_csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep="\t")
train_data = train_csv["question_text"].tolist()


# train_csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep="\t")
# dev_csv = pd.read_csv(open(CIQ_dev_path, encoding="UTF-8"), sep="\t")
# train_data = train_csv["question_text"].tolist()
# train_label = train_csv["target"]
# dev_data = dev_csv["question_text"].tolist()
# dev_label = dev_csv["target"]

#
# tf_idf_vertorize = getTF_IDF_matrix(train_data)
# train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
# dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
#
# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
# svm_clf = SVC()
# voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
# print("集成学习")
# voting_clf.fit(train_tf_idf_vectorize, train_label)
# y_predict = voting_clf.predict(dev_tf_idf_vectorize)
# print(classification_report(dev_label, y_predict))
# print("各个分类器的预测结果：")
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(train_tf_idf_vectorize, train_label)
#     y_pred = clf.predict(dev_tf_idf_vectorize)
#     print(clf.__class__.__name__, accuracy_score(dev_label, y_pred))