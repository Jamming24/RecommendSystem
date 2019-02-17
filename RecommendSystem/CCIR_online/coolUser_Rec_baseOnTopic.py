# encoding:utf-8
# 加载未推荐用户列表，加载用户关注话题列表，加载近4天流行度超过500的ID列表，浏览次数最为这个答案的得分，
# 选出每个话题的前10条记录, 最后通过话题为每个用户进行推荐

import os


def get_NoRecList_User(noreclist_user_file):
    # 返回未推荐用户不感兴趣列表
    norec = open(noreclist_user_file, 'r', encoding='UTF-8')
    noreclist_user = []
    for user in norec:
        noreclist_user.append(user[:32])
    norec.close()
    print(f"noreclist_user:{len(noreclist_user)}")
    norec.close()
    return noreclist_user


def get_UserFollowTopic(user_infos_file, noreclist_user):
    user_followTopic_dict = dict()
    userinfos_object = open(user_infos_file, 'r', encoding='UTF-8')
    for line in userinfos_object:
        tt = line.strip('\n').split('\t')
        userID = tt[0]
        if userID in noreclist_user:
            topics = tt[-1].split(',')
            if len(topics) != 1:
                user_followTopic_dict[userID] = topics
    userinfos_object.close()
    print(f"user_followTopic_dict大小为:{len(user_followTopic_dict)}")
    return user_followTopic_dict


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
        simpleday_Answer_ReadCount_Dict = static_every_testing_set(os.path.join(popularFloder, files_path_dic[key]), round(populary_coefficient, 2))
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


def load_answer_dict(answer_id_dict_file):
    # 根据短ID 找长ID
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        answer_id_dict[tt[0]] = tt[1][:32]
    return answer_id_dict


def loadCandidate(candidate_answer_file):
    candidate = []
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    for line in file_object:
        long_ID = line[:32]
        candidate.append(long_ID)
    print(f"候选集数量：{len(candidate)}")
    return candidate


def load_All_popularAnswer(popularFloder, answer_id_dict, candidate):
    # 返回答案流行度映射字典 ID为长ID 并且都是在候选集合中的答案
    Answer_On_popularity = dict()
    Answer_ReadCount_Dict = computational_popularity(popularFloder)
    for shortanswer in Answer_ReadCount_Dict.keys():
        if shortanswer[0] == 'A' and shortanswer[1:len(shortanswer)] in answer_id_dict.keys():
            longID = answer_id_dict[shortanswer[1:len(shortanswer)]]
            if longID in candidate and Answer_ReadCount_Dict[shortanswer] > 200:
                Answer_On_popularity[longID] = Answer_ReadCount_Dict[shortanswer]
    print(f"Answer_On_popularity数量大小{len(Answer_On_popularity)}")
    return Answer_On_popularity


def get_topicAboutAnswer(answer_infos_all_file, candidate):
    topicAboutAnswer_dict = dict()
    answer_file_object = open(answer_infos_all_file, 'r', encoding='UTF-8')
    for line in answer_file_object:
        tt = line.strip('\n').split('\t')
        answerID = tt[0]
        if answerID in candidate:
            topics = tt[-1].split(',')
            for topic in topics:
                if topic in topicAboutAnswer_dict.keys():
                    temp_answer_list = topicAboutAnswer_dict[topic]
                    temp_answer_list.append(answerID)
                    topicAboutAnswer_dict[topic] = temp_answer_list
                else:
                    answer_list = [answerID]
                    topicAboutAnswer_dict[topic] = answer_list
    answer_file_object.close()
    print(f"topicAboutAnswer_dict大小数量:{len(topicAboutAnswer_dict)}")
    return topicAboutAnswer_dict


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


def get_BaseOnTopic_Rec(UsershowDict, commit_log_Dict, noreclist_user, user_followTopic_dict, topicAboutAnswer_dict, Answer_On_popularity):
    # 查找每个话题对应的问题，参数 用户话题对应字典 话题答案映射字典，答案流行度映射字典，答案ID为长ID
    BaseOnTopic_Rec_dict = dict()
    for userID in noreclist_user:
        show_list = []
        commit_list = []
        if userID in UsershowDict.keys():
            show_list = UsershowDict[userID]
        if userID in commit_log_Dict.keys():
            commit_list = commit_log_Dict[userID]
        if userID in user_followTopic_dict.keys():
            simply_baseOnTopic_Rec = dict()
            user_follow_topic_list = user_followTopic_dict[userID]
            for topic in user_follow_topic_list:
                if topic in topicAboutAnswer_dict.keys():
                    answer_list = topicAboutAnswer_dict[topic]
                    for ans in answer_list:
                        if ans in Answer_On_popularity.keys():
                            simply_baseOnTopic_Rec[ans] = Answer_On_popularity[ans]

            for ll in show_list:
                if ll in simply_baseOnTopic_Rec.keys():
                    simply_baseOnTopic_Rec.pop(ll)
            for com in commit_list:
                if com in simply_baseOnTopic_Rec.keys():
                    simply_baseOnTopic_Rec.pop(com)
            sort_rec = sorted(simply_baseOnTopic_Rec.items(), key=lambda e: e[1], reverse=True)
            sort_simply_baseOnTopic_list = []
            for items in sort_rec:
                sort_simply_baseOnTopic_list.append(items[0])
            BaseOnTopic_Rec_dict[userID] = sort_simply_baseOnTopic_list
    print(f"BaseOnTopic_Rec_dict数量大小:{len(BaseOnTopic_Rec_dict)}")
    return BaseOnTopic_Rec_dict


def Print_BaseOnTopic_Rec(BaseOnTopic_Rec_dict, outfile):
    result_file_object = open(outfile, 'w', encoding='UTF-8')
    for user in BaseOnTopic_Rec_dict.keys():
        result_file_object.write(user+"\t")
        index = 0
        for res in BaseOnTopic_Rec_dict[user]:
            index += 1
            result_file_object.write(res+"@"+str(index)+',')
            if index == 10:
                break
        result_file_object.write("\n")
    result_file_object.close()
    print("未产生推荐列表用户采用基于话题和流行度相结合的方法推荐打印完成")


if __name__ == '__main__':
    user_infos_file = "F:\\CCIR_online\\20180809\\user_infos_20180808_RecSys_flappyBird.txt"
    noreclist_user_file = "F:\\CCIR_online\\20180809\\中间文件\\Item未推荐用户列表_20180808.txt"
    answer_id_dict_file = "F:\\CCIR_online\\20180809\\answer_id_all.dict"
    candidate_answer_file = "F:\\CCIR_online\\20180809\\candidate_online_all.txt"
    answer_infos_all_file = "F:\\CCIR_online\\20180809\\answer_infos_all.txt"
    outfile = "F:\\CCIR_online\\20180809\\中间文件\\noUser_baseonTopic.txt"
    test_set_floder = "F:\\CCIR_online\\test_set"
    commit_log_floder = "F:\\CCIR_online\\commit_log"
    UsershowDict = load_UserShowCard(test_set_floder)
    commit_log_Dict = get_commit_log_Dict(commit_log_floder)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    candidate = loadCandidate(candidate_answer_file)
    noreclist_user = get_NoRecList_User(noreclist_user_file)
    Answer_On_popularity = load_All_popularAnswer(test_set_floder, answer_id_dict, candidate)
    user_followTopic_dict = get_UserFollowTopic(user_infos_file, noreclist_user)
    topicAboutAnswer_dict = get_topicAboutAnswer(answer_infos_all_file, candidate)
    BaseOnTopic_Rec_dict = get_BaseOnTopic_Rec(UsershowDict, commit_log_Dict, noreclist_user, user_followTopic_dict, topicAboutAnswer_dict, Answer_On_popularity)
    Print_BaseOnTopic_Rec(BaseOnTopic_Rec_dict, outfile)
