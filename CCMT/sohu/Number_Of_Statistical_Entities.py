# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 9:56
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Number_Of_Statistical_Entities.py
# @Software: PyCharm
# 统计同一个实体词在不同文章中出现的次数
import re


def get_Entity_Word(train_path):
    Entity_words = set()
    Entity_words_nums = dict()
    p1 = re.compile(r'[[](.*?)[]]', re.S)
    p2 = re.compile(r'["](.*?)["]', re.S)
    # 获取训练集合中的实体词
    file_object = open(train_path, 'r', encoding='UTF-8')
    for line in file_object:
        newID = line.split(",")[0]
        Words = re.findall(p1, line)
        enword = re.findall(p2, str(Words))
        temp = list()
        length = len(enword)
        i = 1
        while i < length:
            temp.append(enword[i])
            Entity_words.add(enword[i])
            if enword[i] in Entity_words_nums.keys():
                newID_list = Entity_words_nums[enword[i]]
                newID_list.append(newID)
                Entity_words_nums[enword[i]] = newID_list
            else:
                newID_list = list()
                newID_list.append(newID)
                Entity_words_nums[enword[i]] = newID_list
            i += 4
        # print(temp)
    file_object.close()
    return Entity_words, Entity_words_nums

train_path = "C:\\Users\\Jamming\\Desktop\\souhu\\sohu_data\\data\\data_v2\\coreEntityEmotion_train.txt"
train_Entity_words, train_Entity_words_nums = get_Entity_Word(train_path=train_path)
print(f"总次数：{len(train_Entity_words)}")
# print(train_Entity_words)
count = 0
for key in train_Entity_words_nums.keys():
    if len(train_Entity_words_nums[key]) >= 500:
        count += 1
        # print(f"实体词：{key},出现次数：{len(train_Entity_words_nums[key])},文章ID：{train_Entity_words_nums[key]}")
        print(f"实体词：{key},出现次数：{len(train_Entity_words_nums[key])}")
print(count)
