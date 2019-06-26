# -*- coding: utf-8 -*-
# @Time    : 2019/6/15 20:18
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : data_process.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

root_path = "E:/Fire2019/评测任务三/"


def standard_generating_data(data_path):
    # 总数：1306118，1225308个0 占比0.93812；80810个1 占比 0.06187  0是正常的 1 是虚假问题
    all_data = pd.read_csv(open(data_path, encoding='UTF-8'))
    print(all_data.shape)
    # print(all_data.describe())
    print(all_data.info())
    print(all_data["target"].value_counts())
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_dev_index, test_index in split.split(all_data, all_data["target"]):
        train_dev_set = all_data.loc[train_dev_index]
        test_set = all_data.loc[test_index]
    train_dev_df = train_dev_set.reset_index().drop("index", axis=1)
    train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, dev_index in train_dev_split.split(train_dev_df, train_dev_df["target"]):
        train_set = train_dev_df.loc[train_index]
        dev_set = train_dev_df.loc[dev_index]
    train_set.to_csv(root_path+"实验文件/train.tsv", sep="\t", index=False, encoding="UTF-8")
    dev_set.to_csv(root_path+"实验文件/dev.tsv", sep="\t", index=False, encoding="UTF-8")
    test_set.to_csv(root_path+"实验文件/test.tsv", sep="\t", index=False, encoding="UTF-8")
    print(f"二分类训练集数据：{train_set.shape[0]}行")
    print(f"二分类开发集数据：{dev_set.shape[0]}行")
    print(f"二分类测试集数据：{test_set.shape[0]}行")

    min_all_data = all_data.loc[0:13061]
    min_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for min_train_dev_index, min_test_index in min_split.split(min_all_data, min_all_data["target"]):
        min_train_dev_set = min_all_data.loc[min_train_dev_index]
        min_test_set = min_all_data.loc[min_test_index]
    min_train_dev_df = min_train_dev_set.reset_index().drop("index", axis=1)
    min_train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for min_train_index, min_dev_index in min_train_dev_split.split(min_train_dev_df, min_train_dev_df["target"]):
        min_train_set = min_train_dev_df.loc[min_train_index]
        min_dev_set = min_train_dev_df.loc[min_dev_index]
    min_train_set.to_csv(root_path+"实验文件/mini_train.tsv", sep="\t", index=False, encoding="UTF-8")
    min_dev_set.to_csv(root_path+"实验文件/mini_dev.tsv", sep="\t", index=False, encoding="UTF-8")
    min_test_set.to_csv(root_path+"实验文件/mini_test.tsv", sep="\t", index=False, encoding="UTF-8")
    print(f"二分类mini训练集数据：{min_train_set.shape[0]}行")
    print(f"二分类mini开发集数据：{min_dev_set.shape[0]}行")
    print(f"二分类mini测试集数据：{min_test_set.shape[0]}行")


def get_ICQtrian_content():
    train_ICQ_df = pd.DataFrame()
    csv_data_all = pd.read_csv(open(root_path + '源文件/train.csv', encoding='UTF-8'))
    csv_data_all.drop(["target"], axis=1, inplace=True)
    ICQ_csv = pd.read_csv(open(root_path+"源文件/CIQ-traindata.tsv", encoding='UTF-8'), sep='\t')
    # 此时的csv_id 是一个Series 可以通过append方法添加到DateFrame中
    csv_id = ICQ_csv["qid"]
    csv_target = ICQ_csv["Label"].tolist()
    print(f"六分类数据总计有{csv_id.count()}行")
    for i in range(csv_id.count()):
        df = csv_data_all[csv_data_all['qid'] == csv_id[i]]
        train_ICQ_df = pd.concat([train_ICQ_df, df])
    print("处理完成")
    print(train_ICQ_df.info())
    train_ICQ_df.insert(2, 'target', csv_target)
    print(train_ICQ_df.info())
    train_ICQ_df.to_csv(root_path+"实验文件/CIQ_traindata_contents.tsv", sep='\t', index=False, encoding='UTF-8')
    print("写入成功")


def standard_generating_CIQ_data():
    # 切分出开发集合
    CIQ_data_csv = pd.read_csv(open(root_path+"实验文件/CIQ_traindata_contents.tsv", encoding="UTF-8"), sep="\t")
    print(CIQ_data_csv.info())
    CIQ_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for CIQ_train_dev_index, CIQ_test_index in CIQ_split.split(CIQ_data_csv, CIQ_data_csv["target"]):
        CIQ_test_set = CIQ_data_csv.loc[CIQ_test_index]
        CIQ_train_dev_set = CIQ_data_csv.loc[CIQ_train_dev_index]
    CIQ_test_set.to_csv(root_path+"实验文件/CIQ_test.tsv", encoding='UTF-8', sep='\t', index=False)
    # 重置索引  并且删除旧的索引行
    CIQ_train_dev_csv = CIQ_train_dev_set.reset_index().drop("index", axis=1)
    CIQ_train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, dev_index in CIQ_train_dev_split.split(CIQ_train_dev_csv, CIQ_train_dev_csv["target"]):
        CIQ_train_set = CIQ_train_dev_csv.loc[train_index]
        CIQ_dev_set = CIQ_train_dev_csv.loc[dev_index]
    CIQ_train_set.to_csv(root_path+"实验文件/CIQ_train.tsv", encoding='UTF-8', index=False, sep='\t')
    CIQ_dev_set.to_csv(root_path+"实验文件/CIQ_dev.tsv", encoding='UTF-8', index=False, sep='\t')
    print(f"CIQ训练集数据：{CIQ_train_set.shape[0]}行")
    print(f"CIQ开发集数据：{CIQ_dev_set.shape[0]}行")
    print(f"CIQ测试集数据：{CIQ_test_set.shape[0]}行")


train_data_path = root_path + '源文件/train.csv'
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
standard_generating_data(data_path=train_data_path)
get_ICQtrian_content()
standard_generating_CIQ_data()