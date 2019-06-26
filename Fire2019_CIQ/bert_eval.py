# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 21:21
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : bert_eval.py
# @Software: PyCharm

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


def load_data_label(data_path):
    data_csv = pd.read_csv(open(data_path, encoding='UTF-8'), sep='\t')
    label_list = data_csv['target']
    print(f"答案总数：{len(label_list)}")
    return label_list


def load_predict_data(predict_path):
    # header=Nones说明没有列名，names=[]表示添加列名
    predict_csv = pd.read_csv(open(predict_path, encoding='UTF-8'), sep='\t', header=None, names=['0', '1', '2', '3', '4', '5'])
    # 返回行的最大值 并添加一个新的列 predict_csv.max(axis=1)返回值 为pandas.core.series.Series
    # predict_csv['max_value'] = predict_csv.max(axis=1)
    # 返回最大值的索引下标
    # predict_csv['max_value_index'] = predict_csv.idxmax(axis=1, skipna=True)
    label_list = predict_csv.idxmax(axis=1, skipna=True).tolist()
    predict_list = pd.to_numeric(label_list)
    print(f"预测答案总数：{len(predict_list)}")
    return predict_list


def eval_one(answer_list, predict_lable):
    # 判断识别1标签的准确率
    total_one = 0
    correct_num = 0
    for index in range(len(answer_list)):
        if answer_list[index] == "1":
            total_one += 1
            if answer_list[index] == predict_lable[index]:
                correct_num += 1
    print(f"1标签总数{total_one}")
    print(f"预测正确总数{correct_num}")
    print(f"预测准确率{correct_num/total_one}")


def eval_zero(answer_list, predict_lable):
    total_zero = 0
    correct_num = 0
    for index in range(len(answer_list)):
        if answer_list[index] == "0":
            total_zero += 1
            if answer_list[index] == predict_lable[index]:
                correct_num += 1
    print(f"0标签总数{total_zero}")
    print(f"预测正确总数{correct_num}")
    print(f"预测准确率{correct_num / total_zero}")


def eval_all(answer_list, predict_lable):
    total = len(answer_list)
    correct_num = 0
    for index in range(len(answer_list)):
        if answer_list[index] == predict_lable[index]:
            correct_num += 1
    print(f"标签总数{total}")
    print(f"预测正确总数{correct_num}")
    print(f"预测准确率{correct_num / total}")


root_path = "E:\\Fire2019\\评测任务三\\"
test_path = root_path + "\\实验文件\\CIQ_dev_predict_one.tsv"
predict_path = root_path+"\\实验文件\\CIQ_dev_predict_one_bert_results.tsv"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
answer_list = load_data_label(test_path)
predict_list = load_predict_data(predict_path)
print("bert实验结果")
print(classification_report(answer_list, predict_list))
test_answer_list = [0, 1, 1, 1, 1, 0]
test_predict_list = [1, 0, 1, 0, 1, 0]
print("打印准确率")
print(precision_score(test_answer_list, test_predict_list))
print("打印召回率")
print(recall_score(test_answer_list, test_predict_list))
print("打印F1")
print(f1_score(test_answer_list, test_predict_list))
# print()
# eval_one(answer_list, predict_lable)
# print()
# eval_zero(answer_list, predict_lable)
# print()
# eval_all(answer_list, predict_lable)
