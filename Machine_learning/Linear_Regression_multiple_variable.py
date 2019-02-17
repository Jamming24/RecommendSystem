# encoding:utf-8

import numpy as np


def load_Data(file_path, begin_line, end_line):
    # begin_line, end_line表示行数范围
    train_value = dict()
    label_value = dict()
    count_line = 0
    file_object = open(file_path, 'r', encoding='UTF-8')
    for line in file_object:
        count_line += 1
        if begin_line <= count_line <= end_line:
            data = line.strip('\n').split('\t')
            label_value[count_line] = float(data[-1])
            temp_data = ['1.0']
            temp_data.extend(data[:-1])
            vector_data = list(map(float, temp_data))
            train_value[count_line] = vector_data
            del temp_data
    file_object.close()
    return train_value, label_value


def Batch_Gradient_Descent(train_value, label_value, learn_rate, parameter_w):
    # 训练数据，标签数据, 学习速率,参数
    # 随机初始化参数向量
    m = len(train_value)
    n = len(parameter_w)
    sum_error = 0.0
    change_num = np.zeros(n)
    for key in train_value.keys():
        coputer_value = np.dot(parameter_w, np.array(train_value[key]))
        real_error = coputer_value - label_value[key]
        for i in range(n):
            change_num[i] += real_error * train_value[key][i]
    parameter_w -= learn_rate * (change_num/m)
    for key in train_value.keys():
        coputer_value = np.dot(parameter_w, np.array(train_value[key]))
        real_error = coputer_value - label_value[key]
        sum_error += np.square(real_error)
    return parameter_w, (sum_error / m)/2


file_path = "E:\\机器学习资料\\Boston_House_Price_Dataset\\housing_data2.txt"
train_Value, label_Value = load_Data(file_path, 1, 300)
nVector = len(train_Value[1])
parameter_w = (np.random.random(nVector)-0.5) * 2
print(parameter_w)
for i in range(10000):
    print(f"第{i}次迭代")
    parameter, sum_error = Batch_Gradient_Descent(train_Value, label_Value, 0.004, parameter_w)
    parameter_w = parameter
    print(parameter)
    print(sum_error)