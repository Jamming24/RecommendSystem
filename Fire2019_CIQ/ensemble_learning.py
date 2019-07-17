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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import clone
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")

from Fire2019_CIQ.brew import Ensemble, EnsembleClassifier
from Fire2019_CIQ.brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from Fire2019_CIQ.brew.combination.combiner import Combiner

root_path = "E:/Fire2019/评测任务三/"
CIQ_train_path = root_path + "实验文件/CIQ_train.tsv"
CIQ_dev_path = root_path + "实验文件/CIQ_dev_test.tsv"
train_csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep="\t")
train_data = train_csv["question_text"].tolist()


train_csv = pd.read_csv(open(CIQ_train_path, encoding="UTF-8"), sep="\t")
dev_csv = pd.read_csv(open(CIQ_dev_path, encoding="UTF-8"), sep="\t")
train_data = train_csv["question_text"].tolist()
train_label = train_csv["target"]
dev_data = dev_csv["question_text"].tolist()
dev_label = dev_csv["target"]


tf_idf_vertorize = getTF_IDF_matrix(train_data)
train_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(train_data), norm='l2')
dev_tf_idf_vectorize = normalize(tf_idf_vertorize.transform(dev_data), norm='l2')
#
log_clf = LogisticRegression(max_iter=10)
svm_clf = SVC(probability=True, decision_function_shape='ovo', kernel="linear", C=1)
knnclf = KNeighborsClassifier(n_neighbors=10)
NB_clf = MultinomialNB(alpha=0.01)
tree_clf = DecisionTreeClassifier(max_depth=3)
rnd_clf = RandomForestClassifier()
# voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('SVM', svm_clf), ("KNN_10", knnclf), ('NB', NB_clf), ('Tree', tree_clf), ('rf', rnd_clf)], voting='soft')
# print("集成学习")
# voting_clf.fit(train_tf_idf_vectorize, train_label)
# y_predict = voting_clf.predict(dev_tf_idf_vectorize)
# print(classification_report(dev_label, y_predict))
# print("各个分类器的预测结果：")
# for clf in (log_clf, svm_clf, knnclf, NB_clf, tree_clf, rnd_clf, voting_clf):
#     clf.fit(train_tf_idf_vectorize, train_label)
#     y_pred = clf.predict(dev_tf_idf_vectorize)
#     print(clf.__class__.__name__, clf.score(train_tf_idf_vectorize, train_label))
#     print(clf.__class__.__name__, accuracy_score(dev_label, y_pred), sep=":")
#
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.6)
# ada_clf.fit(train_tf_idf_vectorize, train_label)
# y_pre = ada_clf.predict(dev_tf_idf_vectorize)
# print("ada精确率:", accuracy_score(y_pred=y_pre, y_true=dev_label))


# Creating Ensemble
ensemble = Ensemble([log_clf, svm_clf, knnclf, NB_clf, tree_clf, rnd_clf])
# ('lr', log_clf), ('SVM', svm_clf), ("KNN_10", knnclf), ('NB', NB_clf), ('Tree', tree_clf), ('rf', rnd_clf)
eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

# Creating Stacking
layer_1 = Ensemble([log_clf, svm_clf, knnclf, NB_clf, tree_clf, rnd_clf])
layer_2 = Ensemble([clone(log_clf)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack)

clf_list = [log_clf, svm_clf, NB_clf, tree_clf, rnd_clf, eclf, sclf]
lbl_list = ['Logistic Regression', 'SVM', 'NB_clf', 'tree_clf', 'rnd_clf', 'Ensemble', 'Stacking']

itt = itertools.product([0, 1, 2, 3, 4, 5, 6, 7], repeat=8)
print("brew----------------")
file_object = open(root_path+"实验文件/commit_result.txt", 'w', encoding='UTF-8')
out_pred = None
for clf, lab, grd in zip(clf_list, lbl_list, itt):
    clf.fit(train_tf_idf_vectorize, train_label)
    y_pred = clf.predict(dev_tf_idf_vectorize)
    print(clf.__class__.__name__, accuracy_score(dev_label, y_pred), sep=":")
    # if clf.__class__.__name__ == "EnsembleStackClassifier":
    #     print("EnsembleStackClassifier输出最终结果")
    #     out_pred = y_pred
    #     print(out_pred.tolist())
    #     for l in out_pred.tolist():
    #         file_object.write(str(l)[0]+'\n')
    #     file_object.close()