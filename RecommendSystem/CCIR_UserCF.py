# encoding:utf-8
import math
from collections import defaultdict


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
            try:
                attr = answer_ques.split("|")
                action_id = attr[0]
                start_time = attr[1]
                end_time = attr[2]
            except Exception:
                print("行为信息部分切分过程中数组下标异常")
            else:
                time_difference = int(end_time) - int(start_time)
                if time_difference > 0:
                    ###########################################
                    # 再这里可以通过计算时间戳的差值进行权重的转换
                    score = 1 / (1 + 1 / (math.exp(time_difference / 60)))
                    answer_ques_score = [action_id, score]
                    behavior_list.append(answer_ques_score)
                    ###########################################
                    # print(action_id)
                    # print(start_time)
                    # print(end_time)
                    if action_id in user_behavior_inverse_table.keys():
                        user_list = user_behavior_inverse_table[action_id]
                        user_list.append(user_Id)
                        user_behavior_inverse_table[action_id] = user_list
                    else:
                        user_list = [user_Id]
                        user_behavior_inverse_table[action_id] = user_list
                else:
                    continue
        # 讲用户行为存到用户行为字典
        user_behavior_dic[user_Id] = behavior_list
    print("use_behavior_dic长度：", len(user_behavior_dic))
    print("user_behavior_inverse_table长度：", len(user_behavior_inverse_table))
    file_object.close()
    return user_behavior_dic, user_behavior_inverse_table


def computerSimilarity_betweenUser(user_behavior_dic, user_behavior_inverse_table, n):
    # 用户行为记录字典, 用户行为记录倒排表字典, 保存前n个最相关用户
    Similarity_betweenUser = dict()
    for user in user_behavior_dic:
        simple_user_behavior = user_behavior_dic[user]
        # simple_user_behavior [('A178557187', 0.710949502625004), ('A195626999', 0.598687660112452),
        simple_user_similarity = dict()
        for user_behavior in simple_user_behavior:
            behavior = user_behavior[0]
            user_list = user_behavior_inverse_table[behavior]
            # 计算完成之后 直接保存于userA最相关的K个用的相似度，其余用于予以舍弃
            # 假设每个用户保留前20个 相关用户 需要消耗 135W x 20 =2700万 项空间 大约是27M整数倍的空间
            for user_id in user_list:
                if user == user_id or len(user_behavior_dic[user_id]) == 0 or \
                        len(simple_user_behavior) == 0 or user_id in simple_user_similarity.keys():
                    continue
                else:
                    score = computerCOS(dict(simple_user_behavior), dict(user_behavior_dic[user_id]))
                    simple_user_similarity[user_id] = score
        max_relevant_Users = sorted(simple_user_similarity.items(), key=lambda e: e[1], reverse=True)[:n]
        # print(user, ">>>>>", len(max_relevant_Users), max_relevant_Users)
        Similarity_betweenUser[user] = max_relevant_Users
    print(f'实际用户数：{len(user_behavior_dic)}')
    print(f'计算用户数：{len(Similarity_betweenUser)}')
    return Similarity_betweenUser


def computerCOS(UserA_dic, UserB_dic):
    numerator = 0.0
    denominator_userA = 0.0
    denominator_userB = 0.0
    for a_key in UserA_dic.keys():
        denominator_userA += math.pow(float(UserA_dic[a_key]), 2)
        if a_key in UserB_dic.keys():
            numerator += float(UserA_dic[a_key]) * float(UserB_dic[a_key])
        else:
            continue
    for b_key in UserB_dic:
        denominator_userB += math.pow(float(UserB_dic[b_key]), 2)
    denominator = math.sqrt(denominator_userA * denominator_userB)
    cos_value = numerator / denominator
    return cos_value


def getRecommend_list(simlarity_user, user_behavior_dic):
    user_Recommend_dic = dict()
    for user_id in user_behavior_dic:
        # 最相关的n个用户
        relevant_Users = simlarity_user[user_id]
        simple_user_list_score = dict()
        for relevant_user in relevant_Users:
            user_score = relevant_user[1]
            user_behavior_score = user_behavior_dic[relevant_user[0]]
            for score in user_behavior_score:
                simple_user_list_score[score[0]] = float(score[1]) * float(user_score)
        sort_simple_user_list = sorted(simple_user_list_score.items(), key=lambda e: e[1], reverse=True)[:100]
        user_Recommend_dic[user_id] = sort_simple_user_list
    return user_Recommend_dic


def print_recommend_list(out_file_commit, out_file_all, recommend_list):
    out_file_commit_object = open(out_file_commit, 'w', encoding='UTF-8')
    out_file_all_object = open(out_file_all, 'w', encoding='UTF-8')
    for user in recommend_list:
        all_line = user + ','
        commit = user + ','
        for index in range(len(user_recommend[user])):
            if index == len(user_recommend[user]) - 1:
                all_line += "(" + user_recommend[user][index][0] + "," + str(user_recommend[user][index][1]) + ")"
                commit += user_recommend[user][index][0]
            else:
                all_line += "(" + user_recommend[user][index][0] + "," + str(user_recommend[user][index][1]) + "),"
                commit += user_recommend[user][index][0] + ","
        out_file_commit_object.write(commit + '\n')
        out_file_all_object.write(all_line + '\n')

    out_file_commit_object.close()
    out_file_all_object.close()


BehaviorFilePath = "E:\\CCIR测试数据\\training_set_Part1.txt"
out_file_commit = "E:\\CCIR测试数据\\commit.txt"
out_file_all = "E:\\CCIR测试数据\\all.txt"
test_user_behavior_dic, test_user_behavior_inverse_table = loadUserBehavior(BehaviorFilePath)
test_Similarity_betweenUser = computerSimilarity_betweenUser(test_user_behavior_dic, test_user_behavior_inverse_table,
                                                             5)
user_recommend = getRecommend_list(test_Similarity_betweenUser, test_user_behavior_dic)
print_recommend_list(out_file_commit, out_file_all, user_recommend)
