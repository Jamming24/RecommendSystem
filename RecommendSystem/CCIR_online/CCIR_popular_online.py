# encoding:utf-8
# 基于流行度的推荐算法

import os


def computational_popularity(popularFloder):
    # 计算每个被阅读的答案近四天的流行分布
    # 26号的次数*0.7 +27号的次数*0.8+28号的次数*0.9+29号的次数*1
    Answer_ReadCount_Dict = dict()
    files = os.listdir(popularFloder)
    files_path_dic = dict()
    for f in files:
        files_path_dic[f[17:20]] = f
    sort_list = sorted(files_path_dic.keys(), reverse=True)
    populary_coefficient = 1.0
    for key in sort_list:
        print(f"开始进行{key}日流行度统计")
        simpleday_Answer_ReadCount_Dict = static_every_testing_set(os.path.join(popular_floder, files_path_dic[key]), round(populary_coefficient, 2))
        if len(Answer_ReadCount_Dict) == 0:
            Answer_ReadCount_Dict = simpleday_Answer_ReadCount_Dict.copy()
        else:
            for ansid in simpleday_Answer_ReadCount_Dict.keys():
                if ansid in Answer_ReadCount_Dict.keys():
                    Answer_ReadCount_Dict[ansid] += simpleday_Answer_ReadCount_Dict[ansid]
                else:
                    Answer_ReadCount_Dict[ansid] = simpleday_Answer_ReadCount_Dict[ansid]
        populary_coefficient = populary_coefficient - 0.1
    return Answer_ReadCount_Dict


def static_every_testing_set(every_test_file, populary_coefficient):
    # 返回每个问题或者话题ID被浏览的次数 被展示但是为被用户阅读的问题不记录次数
    print("流行度系数：" + str(populary_coefficient))
    Answer_ReadCount_Dict = dict()
    file_object = open(every_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        if len(temp) > 2:
            for answer in temp[2].split(","):
                if len(answer) != 0:
                    t = answer.split("|")
                    answerID = t[0]
                    # 展示时间
                    # start_time = t[1]
                    # 用户阅读时间
                    end_time = t[2]
                    if int(end_time) != 0:
                        if answerID in Answer_ReadCount_Dict.keys():
                            Answer_ReadCount_Dict[answerID] += 1
                        else:
                            Answer_ReadCount_Dict[answerID] = 1
    file_object.close()
    for ans in Answer_ReadCount_Dict.keys():
        Answer_ReadCount_Dict[ans] = Answer_ReadCount_Dict[ans] * populary_coefficient
    print(f"Answer_ReadCount_Dict数量:{len(Answer_ReadCount_Dict)}")
    return Answer_ReadCount_Dict


def get_NoRecList_User(noreclist_user_file):
    # 返回未推荐用户不感兴趣字典
    norec = open(noreclist_user_file, 'r', encoding='UTF-8')
    noreclist_user = []
    for user in norec:
        noreclist_user.append(user[:32])
    norec.close()
    print(f"noreclist_user:{len(noreclist_user)}")
    norec.close()
    return noreclist_user


def load_Answer_stamp(answerinfo_file):
    file_object = open(answerinfo_file, 'r', encoding='UTF-8')
    answerID_stamp = dict()
    # oneday = 0
    # twoday = 0
    # threeday = 0
    # fourday = 0
    # other = 0
    for line in file_object:
        tt = line.split("\t")
        answerID = tt[0]
        time_stamp = int(tt[6])
        answerID_stamp[answerID] = time_stamp
    file_object.close()
    return answerID_stamp


def load_answer_dict(answer_id_dict_file):
    # 根据短ID 找长ID
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        answer_id_dict[tt[0]] = tt[1][:32]
    return answer_id_dict


def load_question_dict(question_id_dict_file):
    question_id_dict = dict()
    file_object = open(question_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        question_id_dict[tt[0]] = tt[1][:32]
    return question_id_dict


def loadCandidate(candidate_answer_file):
    candidate = []
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    for line in file_object:
        long_ID = line[:32]
        candidate.append(long_ID)
    print(f"候选集数量：{len(candidate)}")
    return candidate


def get_Popular_Rec(noreclist_user, Answer_ReadCount_Dict, answerID_stamp, answer_id_dict, UsershowDict):
    popular_rec_dict = dict()
    for answer in Answer_ReadCount_Dict.keys():
        shortID = answer[1:len(answer)]
        if answer[0] == 'A' and shortID in answer_id_dict.keys():
            longID = answer_id_dict[shortID]
            if longID in answerID_stamp.keys():
                timeStamp = answerID_stamp[longID]
                if timeStamp < 1532361600:
                    # 过滤掉7月24日之前的答案
                    Answer_ReadCount_Dict[answer] = 0
    already_sort = sorted(Answer_ReadCount_Dict.items(), key=lambda e: e[1], reverse=True)
    # for i in already_sort:
    #     print(i[0]+","+str(i[1]))
    for user in noreclist_user:
        simply_rec_list = []
        if user in UsershowDict.keys():
            showItems = UsershowDict[user]
            for item in already_sort:
                if item[0] not in showItems:
                    simply_rec_list.append(item[0])
        else:
            for item in already_sort:
                simply_rec_list.append(item[0])
        popular_rec_dict[user] = simply_rec_list
    print("排序完成")
    return popular_rec_dict


def get_Popular_Rec_all(testID, Answer_ReadCount_Dict , UsershowDict):
    small_Answer_ReadCount_Dict = dict()
    for k in Answer_ReadCount_Dict.keys():
        temp = Answer_ReadCount_Dict[k]
        if temp > 200:
            small_Answer_ReadCount_Dict[k] = temp
    print(f"small_Answer_ReadCount_Dict大小为:{len(small_Answer_ReadCount_Dict)}")
    popular_rec_dict = dict()
    for user in testID:
        simply_user_rec = []
        if user in UsershowDict.keys():
            simply_user_showlist = UsershowDict[user]
            popular_dict = small_Answer_ReadCount_Dict.copy()
            for item in popular_dict.keys():
                if item in simply_user_showlist:
                    popular_dict[item] = 0
            already_sort = sorted(popular_dict.items(), key=lambda e: e[1], reverse=True)
            popular_dict.clear()
            for index in range(0, 200):
                simply_user_rec.append(already_sort[index][0])
        else:
            popular_dict = Answer_ReadCount_Dict.copy()
            already_sort = sorted(popular_dict.items(), key=lambda e: e[1], reverse=True)
            popular_dict.clear()
            for index in range(0, 200):
                simply_user_rec.append(already_sort[index][0])
        popular_rec_dict[user] = simply_user_rec
        # print(simply_user_rec)
    return popular_rec_dict


def PrintPopular_Rec(n, popular_rec_dict, answer_id_dict, candidate, outfile, commit_log_Dict):
    result_file_object = open(outfile, 'w', encoding='UTF-8')
    for user in popular_rec_dict.keys():
        result_file_object.write(user + '\t')
        max_relevant = popular_rec_dict[user]
        if user not in commit_log_Dict.keys():
            index = 0
            for item in max_relevant:
                if item[1:len(item)] in answer_id_dict.keys():
                    answerID = answer_id_dict[item[1:len(item)]]
                    if answerID in candidate:
                        index += 1
                        result_file_object.write(answerID + "@" + str(index) + ',')
                        if index == n:
                            break
            result_file_object.write('\n')
        else:
            commit_log_set = commit_log_Dict[user]
            index = 0
            for item in max_relevant:
                if item[1:len(item)] in answer_id_dict.keys():
                    answerID = answer_id_dict[item[1:len(item)]]
                    if answerID in candidate and answerID not in commit_log_set:
                        index += 1
                        result_file_object.write(answerID + "@" + str(index) + ',')
                        if index == n:
                            break
            result_file_object.write('\n')
    result_file_object.close()
    print("结果输出完成")


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


if __name__ == '__main__':
    every_test_file = "E:\\CCIR_online\\20180809\\testing_set_20180808_RecSys_flappyBird.txt"
    noreclist_user_file = "E:\\CCIR_online\\20180809\\中间文件\\Item未推荐用户列表_20180808.txt"
    answerinfo_file = "E:\\CCIR_online\\20180809\\answer_infos_all.txt"
    answer_id_dict_file = "E:\\CCIR_online\\20180809\\answer_id_all.dict"
    outfile = "E:\\CCIR_online\\20180809\\中间文件\\popular_reclist.csv"
    candidate_answer_file = "F:\\CCIR_online\\20180809\\candidate_online_all.txt"
    test_set_floder = "E:\\CCIR_online\\test_set"
    commit_log_floder = "E:\\CCIR_online\\commit_log"
    #######################################
    # 为未产生推荐列表的用户推荐流行答案
    Answer_ReadCount_Dict = static_every_testing_set(every_test_file, 1.0)
    commit_log_Dict = get_commit_log_Dict(commit_log_floder)
    UsershowDict = load_UserShowCard(test_set_floder)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    answerID_stamp = load_Answer_stamp(answerinfo_file)
    noreclist_user = get_NoRecList_User(noreclist_user_file)
    candidate = loadCandidate(candidate_answer_file)
    popular_rec_dict = get_Popular_Rec(noreclist_user, Answer_ReadCount_Dict, answerID_stamp, answer_id_dict, UsershowDict)
    PrintPopular_Rec(10, popular_rec_dict, answer_id_dict, candidate, outfile, commit_log_Dict)
    #######################################
    # 流行度推荐更新
    # 为所以用户推荐流行答案
    # commit_log_floder = "F:\\CCIR_online\\commit_log"
    # commit_log_Dict = get_commit_log_Dict(commit_ log_floder)
    # popular_floder = "F:\\CCIR_online\\test_set"
    # candidate_answer_file = "F:\\CCIR_online\\20180730\\candidate_online_all.txt"
    # outfile = "F:\\CCIR_online\\20180730\\中间文件\\0730_popular_reclist.csv"
    # answer_id_dict_file = "F:\\CCIR_online\\20180730\\answer_id_all.dict"
    # # answer_id_dict = load_answer_dict(answer_id_dict_file)
    # Answer_ReadCount_Dict = computational_popularity(popular_floder)
    # sor = sorted(Answer_ReadCount_Dict.items(), key=lambda e: e[1], reverse=True)
    # for key in sor:
    #     if key[1] > 340:
    #         print(key[0] + "," + str(round(key[1], 2)))
    # candidate = loadCandidate(candidate_answer_file)
    # UsershowDict, testID = get_UsershowDict(every_test_file)
    # popular_rec_dict = get_Popular_Rec_all(testID, Answer_ReadCount_Dict, UsershowDict)
    # PrintPopular_Rec(10, popular_rec_dict, answer_id_dict, candidate, outfile, commit_log_Dict)
    ##########################################
