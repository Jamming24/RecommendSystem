# encoding:utf-8

import os
import math


def load_Item_Similary(similaryFloder):
    Item_Similary_dict = dict()
    file_list = os.listdir(similaryFloder)
    for fileName in file_list:
        filePath = os.path.join(similaryFloder, fileName)
        file_object = open(filePath, 'r', encoding='UTF-8')
        for line in file_object:
            simple_ItemSimilary_list = []
            t = line.split('\t')
            Item_ID = t[0]
            revelent_answer = t[1].strip('\n').split(',')
            for item in revelent_answer[:len(revelent_answer)-1]:
                simple_ItemSimilary_list.append([item.split(':')[0], float(item.split(':')[1])])
            Item_Similary_dict[Item_ID] = simple_ItemSimilary_list
        file_object.close()
    print(f"相似度字典大小{len(Item_Similary_dict)}")
    return Item_Similary_dict


def load_every_testing_set(every_test_file, test_set_floder):
    User_Behavior_Dict = dict()
    file_object = open(every_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        userID = temp[0]
        answer_list = []
        for answer in temp[2].split(","):
            if len(answer) != 0:
                t = answer.split("|")
                start_time = t[1]
                end_time = t[2]
                if int(end_time) != 0:
                    time_difference = int(start_time) - int(end_time)
                    if time_difference > 0:
                        answer_list.append([t[0], 1.0])
                    else:
                        score = 1 / (1 + 1 / (math.exp(time_difference / 60)))
                        # print(time_difference/60, ',', score)
                        answer_list.append([t[0], score])
        if len(answer_list) != 0:
            User_Behavior_Dict[userID] = answer_list
    file_object.close()
    print(f"test用户数量{len(User_Behavior_Dict)}")

    userID_set = User_Behavior_Dict.keys()
    files = os.listdir(test_set_floder)
    for file in files:
        simply_path = os.path.join(test_set_floder, file)
        old_file_object = open(simply_path, 'r', encoding='UTF-8')
        for old_line in old_file_object:
            old_temp = old_line.split("\t")
            temp_userID = old_temp[0]
            if temp_userID in userID_set:
                temp_answer_set = set()
                temp_answer_list = User_Behavior_Dict[temp_userID]
                for u in temp_answer_list:
                    temp_answer_set.add(u[0])
                old_num = len(temp_answer_set)
                old_Answers = old_temp[2].split(",")
                for old_answer in old_Answers:
                    if len(old_answer) != 0:
                        old_t = old_answer.split("|")
                        old_start_time = old_t[1]
                        old_end_time = old_t[2]
                        if int(old_end_time) != 0 and old_t[0] not in temp_answer_set:
                            old_time_difference = int(old_start_time) - int(old_end_time)
                            if old_time_difference > 0:
                                temp_answer_list.append([old_t[0], 1.0])
                            else:
                                old_score = 1 / (1 + 1 / (math.exp(old_time_difference / 60)))
                                temp_answer_list.append([old_t[0], old_score])
                        else:
                            continue
                User_Behavior_Dict[temp_userID] = temp_answer_list
                print(f"{temp_userID}新增行为数量{len(temp_answer_list) - old_num}")
        old_file_object.close()
    return User_Behavior_Dict


def get_User_Recommond_Dict(UsershowDict, commit_log_Dict, User_Behavior_Dict, Item_Similary_dict, revers_answer_id_dict, k):
    # k值表示为与用户喜欢的物品最相关的k个物品
    User_Recommond_Dict =dict()
    for userID in User_Behavior_Dict:
        simiply_user_rec = dict()
        simply_show_answer_list = []
        simply_log_list = set()
        if userID in UsershowDict.keys():
            simply_show_answer_list = UsershowDict[userID]
        if userID in commit_log_Dict.keys():
            temp_log_set = commit_log_Dict[userID]
            temp_log_set.remove("")
            for commit in temp_log_set:
                simply_log_list.add('A' + revers_answer_id_dict[commit])
        for similaryItem in User_Behavior_Dict[userID]:
            # 用户行为 指文章ID 和 得分
            aid = similaryItem[0]
            score = similaryItem[1]
            if aid in Item_Similary_dict.keys():
                similaryList = Item_Similary_dict[aid][:k]
                for everyItem in similaryList:
                    # 待推荐物品
                    itemName = everyItem[0]
                    itemScore = everyItem[1]
                    if itemName in simply_show_answer_list or itemName in simply_log_list:
                        continue
                    else:
                        if itemName in simiply_user_rec:
                            temp_finalScore = simiply_user_rec[itemName]
                            finalScore = temp_finalScore + (score * itemScore)
                            simiply_user_rec[itemName] = finalScore
                        else:
                            simiply_user_rec[itemName] = score * itemScore
        simply_log_list.clear()
        simply_show_answer_list.clear()
        if len(simiply_user_rec) != 0:
            User_Recommond_Dict[userID] = simiply_user_rec
    return User_Recommond_Dict


def Print_User_Recommond_Dict(n, result_file, result_file_withScore, User_Recommond_Dict, answer_id_dict, candidate):
    result_file_withScore_object = open(result_file_withScore, 'w', encoding='UTF-8')
    result_file_object = open(result_file, 'w', encoding='UTF-8')
    for key in User_Recommond_Dict.keys():
        result_file_withScore_object.write(key+'\t')
        result_file_object.write(key+'\t')
        max_relevant = sorted(User_Recommond_Dict[key].items(), key=lambda e: e[1], reverse=True)[:n]
        index = 0
        for item in max_relevant:
            answerID = answer_id_dict[item[0][1:len(item[0])]]
            # print(answerID)
            if answerID in candidate:
                result_file_withScore_object.write(answerID+":"+str(item[1])+",")
                index += 1
                result_file_object.write(answerID+"@"+str(index)+',')
        result_file_withScore_object.write('\n')
        result_file_object.write('\n')
    result_file_object.close()
    result_file_withScore_object.close()
    print("结果输出完成")


def loadCandidate(candidate_answer_file):
    candidate = []
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    for line in file_object:
        long_ID = line[:32]
        candidate.append(long_ID)
    print(f"候选集数量：{len(candidate)}")
    return candidate


def load_answer_dict(answer_id_dict_file):
    # 根据短ID 找长ID
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        answer_id_dict[tt[0]] = tt[1][:32]
    return answer_id_dict


def get_NoReclist_user(every_test_file, result_file, outfile):
    test_object = open(every_test_file, 'r', encoding='UTF-8')
    result_object = open(result_file, 'r', encoding='UTF-8')
    out_object = open(outfile, 'w', encoding='UTF-8')
    test_ID = set()
    result_ID = set()
    for line in test_object:
        userID = line.split('\t')[0]
        test_ID.add(userID)
    print(f"testID大小:{len(test_ID)}")
    for res in result_object:
        result_ID.add(res.split('\t')[0])
    print(f"result_ID大小:{len(result_ID)}")
    test_ID = test_ID.difference(result_ID)
    print(f"未推荐用户数量:{len(test_ID)}")
    for k in test_ID:
        out_object.write(k+"\n")
    test_object.close()
    result_object.close()
    out_object.close()


def load_UserShowCard(test_set_floder):
    # 加载用户阅读和展示过的内容
    files = os.listdir(test_set_floder)
    UsershowDict = dict()
    for file in files:
        simply_path = os.path.join(test_set_floder, file)
        file_object = open(simply_path, 'r', encoding='UTF-8')
        for line in file_object:
            temp = line.split("\t")
            userID = temp[0]
            show_answer_set = set()
            for answer in temp[2].split(","):
                if len(answer) != 0:
                    t = answer.split("|")
                    show_answer_set.add(t[0])
            UsershowDict[userID] = show_answer_set
        file_object.close()
    print(f"UsershowDict大小:{len(UsershowDict)}")
    return UsershowDict


def get_commit_log_Dict(commit_log_floder):
    commit_log_Dict = dict()
    files = os.listdir(commit_log_floder)
    for file in files:
        filePath = os.path.join(commit_log_floder, file)
        file_object = open(filePath, 'r', encoding='UTF-8')
        for line in file_object:
            commit_answer_set = set()
            userID = line.split('\t')[0]
            answerID = line.split('\t')[1].strip('\n').split(',')
            for ans in answerID:
                if userID in commit_log_Dict.keys():
                    temp_set = commit_log_Dict[userID]
                    temp_set.add(ans[:32])
                    commit_log_Dict[userID] = temp_set
                else:
                    commit_answer_set.add(ans[:32])
                    commit_log_Dict[userID] = commit_answer_set
    print(f"commit_log_Dict大小:{len(commit_log_Dict)}")
    return commit_log_Dict


def load_revers_answer_dict(answer_id_dict_file):
    # 根据长ID找短ID
    revers_answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        revers_answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    return revers_answer_id_dict


if __name__ == '__main__':
    similaryFloder = "E:\\CCIR_online\\20180809\\中间文件\\Item_Similary"
    every_test_file = "E:\\CCIR_online\\20180809\\testing_set_20180808_RecSys_flappyBird.txt"
    result_file = "E:\\CCIR_online\\20180809\\中间文件\\item_result_file_20180808.csv"
    result_file_withScore = "E:\\CCIR_online\\20180809\\中间文件\\item_result_withScore_file_20180808.csv"
    candidate_answer_file = "E:\\CCIR_online\\20180809\\candidate_online_all.txt"
    answer_id_dict_file = "E:\\CCIR_online\\20180809\\answer_id_all.dict"

    test_set_floder = "E:\\CCIR_online\\test_set"
    commit_log_floder = "E:\\CCIR_online\\commit_log"
    ####################################
    # 获取ItemCF推荐结果
    candidate = loadCandidate(candidate_answer_file)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    revers_answer_id_dict = load_revers_answer_dict(answer_id_dict_file)
    UsershowDict = load_UserShowCard(test_set_floder)
    commit_log_Dict = get_commit_log_Dict(commit_log_floder)
    User_Behavior_Dict = load_every_testing_set(every_test_file, test_set_floder)
    count = 0
    for user in UsershowDict.keys():
        if len(UsershowDict[user]) < 5:
            count += 1
    print(count)
    Item_Similary_dict = load_Item_Similary(similaryFloder)
    User_Recommond_Dict = get_User_Recommond_Dict(UsershowDict, commit_log_Dict, User_Behavior_Dict, Item_Similary_dict, revers_answer_id_dict, 150)
    Print_User_Recommond_Dict(10, result_file, result_file_withScore, User_Recommond_Dict, answer_id_dict, candidate)
    ####################################
    # 打印未产生推荐列表的用户
    outfile = "E:\\CCIR_online\\20180809\\中间文件\\Item未推荐用户列表_20180809.txt"
    get_NoReclist_user(every_test_file, result_file, outfile)
