# coding=utf-8

import os
import pandas as pd


def ReadTrainingSet():
    trainSetFolder = "E:\\CCIR测试数据"
    Files = os.listdir(trainSetFolder)
    for f in Files:
        if 24 > len(f) > 21:
            filePath = os.path.join(trainSetFolder, f)
            print(filePath)
            # 参数：names=["用户ID", "交互数", "交互内容", "搜素行为数","搜素内容", "阅读时间", "文章类型", "文章ID"],
            data = pd.read_csv(filePath, delimiter="\t", header=None, encoding="UTF-8")
            print(data[:5])


ReadTrainingSet()
