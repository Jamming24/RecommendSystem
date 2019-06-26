# -*- coding:utf-8 -*-

import os
import pandas as pd


def get_identifyUser_record(Metro_train_Floder, Metro_train_identifyUser_Floder):
    for root, dirs, files in os.walk(top=Metro_train_Floder, topdown=True):
        for file_name in files:
            metro_train_file_path = os.path.join(root, file_name)
            metro_train_identifyUser_file = os.path.join(Metro_train_identifyUser_Floder, file_name)
            data = pd.read_csv(metro_train_file_path, engine='python', usecols=["time", "lineID", "stationID", "deviceID", "status", "userID", "payType"])
            identifyUser_data = data[data['payType'] < 3]
            identifyUser_data.to_csv(metro_train_identifyUser_file, index=False)
            print(file_name, "处理完成")
    print(Metro_train_Floder, "处理完成")


if __name__ == '__main__':

    len([1,2,3])
    Metro_train_Floder = "E:\\地铁乘客流量预测\\Metro_train"
    Metro_train_identifyUser_Floder = "E:\地铁乘客流量预测\Metro_train_identifyUser"
    # 将有唯一标识的用户筛选出来
    # get_identifyUser_record(Metro_train_Floder=Metro_train_Floder,
    # Metro_train_identifyUser_Floder=Metro_train_identifyUser_Floder)

    # 将所有用户ID筛选出来，以便做映射，顺便观察每天的重复用户占多少
    for root, dirs, files in os.walk(top=Metro_train_identifyUser_Floder, topdown=True):
        flag = False
        for file_name in files:
            print("正在加载文件:", file_name)
            metro_train_file_path = os.path.join(root, file_name)
            data = pd.read_csv(metro_train_file_path, engine='python', usecols=["userID"])
            data = pd.DataFrame(data)
            if not flag:
                flag = True
                last_data = data
                last_data_name = file_name
                continue
            else:
                data_set = set()
                last_data_set = set()
                for index, row in data.iterrows():
                    data_set.add(row["userID"])
                for index, row in last_data.iterrows():
                    last_data_set.add(row["userID"])
                merge_data = data_set.intersection(last_data_set)
                print(file_name, "记录数：", data.shape[0], ">>>", data.shape[1])
                print(file_name, "记录数set：", len(data_set), ">>>", data.shape[1])
                print(last_data_name, "记录数：", last_data.shape[0], ">>>", last_data.shape[1])
                print(last_data_name, "记录数set：", len(last_data_set), ">>>", last_data.shape[1])
                print(file_name, ">>>>>", last_data_name, "交集数量:", len(merge_data))
                last_data = data
                last_data_name = file_name

