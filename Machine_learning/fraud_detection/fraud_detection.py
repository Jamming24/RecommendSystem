# -*- coding: utf-8 -*-
# @Time    : 2019/8/25 20:58
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : fraud_detection.py
# @Software: PyCharm

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

root_path = "E:/Kaggle/ieee-fraud-detection/"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def load_data():
    train_data_csv = pd.read_csv(root_path+"train_transaction.csv")
    # print(data_csv.info())
    # RangeIndex: 590540 entries, 0 to 590539
    # Columns: 394 entries, TransactionID to V339
    # dtypes: float64(376), int64(4), object(14)
    # memory usage: 1.7+ GB
    # print(data_csv["isFraud"].value_counts())
    # 0    569877 占比96.5%
    # 1     20663
    # print(train_data_csv.describe())
    isFraud_data = train_data_csv[train_data_csv["isFraud"] > 0]
    # print(isFraud_data)
    isFraud_data.to_csv(root_path+"train_isFraud_data.csv", sep=',', index=False)


def train():
    train_data = train_data_csv[["TransactionAmt", "card1", "card2", "addr1", "card5", "C1"]].fillna(train_data_csv[["TransactionAmt", "card1", "card2", "addr1", "card5", "C1"]].mean())
    train_label = train_data_csv["isFraud"]
    lg = LogisticRegression()
    lg.fit(train_data, train_label)
    test_csv = pd.read_csv(root_path+"test_transaction.csv")
    test_data = test_csv[["TransactionAmt", "card1", "card2", "addr1", "card5", "C1"]].fillna(test_csv[["TransactionAmt", "card1", "card2", "addr1", "card5", "C1"]].mean())
    y_predict = lg.predict_proba(test_data)
    test_ID = test_csv["TransactionID"]
    y_predict = pd.DataFrame(y_predict[1])
    result = test_ID.append(y_predict)
    print(result)
    # out = pd.merge(test_ID, )
    # out.to_csv(root_path+"scond_submission.csv", sep=",")


# load_data()
root_path = "E:/Fire2019/评测任务三/"
CIQ_dev_path = root_path + "实验文件/CIQ_dev.tsv"
dev_csv = pd.read_csv(open(CIQ_dev_path, encoding="UTF-8"), sep="\t")
dev_label = dev_csv["target"]
skf = model_selection.StratifiedKFold([1, 2, 1, 0, 1, 0], 5)