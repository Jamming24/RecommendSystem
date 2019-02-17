# encoding:utf-8
import math
import multiprocessing
import os
import numpy.linalg as li
import numpy as np


def loadUserBehavior(behaviorFilePath):
    # 用户行为记录字典
    user_behavior_dic = dict()
    # 用户行为记录倒排表字典
    user_behavior_inverse_table = dict()
    file_object = open(behaviorFilePath, 'r', encoding='UTF-8')
    behaviorData = file_object.readlines()
    for data in behaviorData:
        behavior_list = list()
        datas = data.split('\t')
        # 用户ID
        user_Id = datas[0]
        # 用户行为数量
        # behavior_num = datas[1]
        # 用户行为记录
        behavior_content = datas[2]
        # 建立用户行为倒排表
        answer_questions = behavior_content.split(',')
        for answer_ques in answer_questions:
            attr = answer_ques.split("|")
            if len(attr) == 3:
                action_id = attr[0]
                start_time = attr[1]
                end_time = attr[2]
                if int(end_time) != 0:
                    time_difference = int(end_time) - int(start_time)
                    ###########################################
                    # 再这里可以通过计算时间戳的差值进行权重的转换
                    # jaccard 系数
                    behavior_list.append(action_id)
                    ###########################################
                    if action_id in user_behavior_inverse_table.keys():
                        user_list = user_behavior_inverse_table[action_id]
                        user_list.append(user_Id)
                        user_behavior_inverse_table[action_id] = user_list
                    else:
                        user_list = [user_Id]
                        user_behavior_inverse_table[action_id] = user_list
                # 讲用户行为存到用户行为字典
        user_behavior_dic[user_Id] = behavior_list
    print("use_behavior_dic长度：", len(user_behavior_dic))
    print("user_behavior_inverse_table长度：", len(user_behavior_inverse_table))
    file_object.close()
    return user_behavior_dic, user_behavior_inverse_table


def computerSimilarity_betweenUser(user_behavior_dic, user_behavior_inverse_table, n, out_Floder):
    # 用户行为记录字典, 用户行为记录倒排表字典, 保存前n个最相关用户
    # 使用进程池技术
    pool = multiprocessing.Pool(processes=6)
    user_list_temp = []
    file_count = 1
    for user in user_behavior_dic.keys():
        # simple_user_behavior [('A178557187', 0.710949502625004), ('A195626999', 0.598687660112452),
        user_list_temp.append(user)
        if len(user_list_temp) == 10000:
            part_user_list = user_list_temp.copy()
            out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
            temp_user_behavior_dic = user_behavior_dic.copy()
            temp_user_behavior_inverse_table = user_behavior_inverse_table.copy()
            pool.apply_async(userThreadCompute, args=(part_user_list, temp_user_behavior_dic, temp_user_behavior_inverse_table, n, out_file))
            # userThreadCompute(user_list_temp, user_behavior_dic, user_behavior_inverse_table, n, out_file)
            user_list_temp.clear()
            file_count += 1

    print('最后一部分的用户数：', len(user_list_temp))
    # userThreadCompute(user_list_temp, user_behavior_dic, user_behavior_inverse_table, n, out_file)
    out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
    pool.apply_async(userThreadCompute, args=(user_list_temp, user_behavior_dic, user_behavior_inverse_table, n, out_file))
    pool.close()
    pool.join()
    print(f'实际用户数：{len(user_behavior_dic)}')


def userThreadCompute(part_user_list, user_behavior_dic, user_behavior_inverse_table, n, out_file):
    print(f'线程{out_file}计算开始')
    max_relevant_Users_file = open(out_file, 'w', encoding='UTF-8')
    #   计算完成以后直接输出到单个文件进行保存
    for user in part_user_list:
        simple_user_behavior = user_behavior_dic[user]
        simple_user_similarity = dict()
        for user_behavior in simple_user_behavior:
            behavior_id = user_behavior
            user_list = user_behavior_inverse_table[behavior_id]
            # 计算完成之后 直接保存于userA最相关的K个用的相似度，其余用于予以舍弃
            # 假设每个用户保留前20个 相关用户 需要消耗 135W x 20 =2700万 项空间 大约是27M整数倍的空间
            for user_id in user_list:
                if user == user_id or len(user_behavior_dic[user_id]) == 0 or \
                        len(simple_user_behavior) == 0 or user_id in simple_user_similarity.keys():
                    continue
                else:
                    score = computerJaccard(simple_user_behavior, user_behavior_dic[user_id])
                    simple_user_similarity[user_id] = score
        max_relevant_Users = sorted(simple_user_similarity.items(), key=lambda e: e[1], reverse=True)[:n]
        # print(user, ">>>>>", len(max_relevant_Users), max_relevant_Users)
        max_relevant_Users_file.write(f"用户ID:{user},")
        for i in range(len(max_relevant_Users)):
            line = str(i + 1)+":("+str(max_relevant_Users[i][0])+","+str(max_relevant_Users[i][1])+"),"
            max_relevant_Users_file.write(line)
        max_relevant_Users_file.write("\n")
    max_relevant_Users_file.close()
    print(f'线程{out_file}计算完成')


def computerJaccard(UserA_list, UserB_list):
    numerator = len(list(set(UserA_list).intersection(set(UserB_list))))
    denominator = len(list(set(UserA_list).union(set(UserB_list))))
    Jaccard_value = numerator / denominator
    # print(numerator, ">>>>>>", denominator, ">>>>>>>>", Jaccard_value)
    return Jaccard_value


if __name__ == '__main__':
    # 本次实验将使用jacard系数做相似度计算
    BehaviorFilePath = "E:\\CCIR\\small_testing_set.txt"
    out_Floder = "E:\\CCIR\\Similarity_betweenUser_Floder"
    test_user_behavior_dic, test_user_behavior_inverse_table = loadUserBehavior(BehaviorFilePath)
    computerSimilarity_betweenUser(test_user_behavior_dic, test_user_behavior_inverse_table, 100, out_Floder)
