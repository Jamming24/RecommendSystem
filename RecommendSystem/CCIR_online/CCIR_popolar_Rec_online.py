# encoding:utf-8
# 基于流行度的推荐算法 扩展
# 全体最热门推荐7条， 当天最热门的推荐3个
import os


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


def get_testID(evaryday_test):
    testID = []
    file_object = open(evaryday_test, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        userID = temp[0]
        testID.append(userID)
    file_object.close()
    print(f"testID大小:{len(testID)}")
    return testID


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

# 全体最热门推荐7条， 当天最热门的推荐3个


def popular_rec(hot_file, answer_id_dict, candidate, UsershowDict, commit_log_Dict, testID):
    # 取最热门的前1000个进行过滤 要求必须在候选集合中，（这个最先处理），然后过滤点用户展示和阅读过的
    # 最后过滤掉为用户推荐过的, 最后每个用户取前10条进行返回
    popular_rec_dict = dict()
    hot_file_object = open(hot_file, 'r', encoding='UTF-8')
    hot_list = []
    for line in hot_file_object:
        hot_list.append(line.split(',')[0])
    hot_list = hot_list[:1000]
    print(f"初试流行度候选集合大小hot_list初试数量{len(hot_list)}")
    for item in hot_list:
        ID_num = item[1:len(item)]
        if ID_num not in answer_id_dict.keys() or item[0] != 'A' or answer_id_dict[ID_num] not in candidate:
            hot_list.remove(item)
    print(f"过滤掉候选集合之外的答案hot_list剩余数量{len(hot_list)}")
    for user in testID:
        simply_user_rec_list = hot_list.copy()
        if user in UsershowDict.keys():
            simply_show_list = UsershowDict[user]
            for recitem in simply_user_rec_list:
                if recitem in simply_show_list:
                    simply_user_rec_list.remove(recitem)
            # print(f"去掉用户{user}浏览行为simply_user_rec_list的大小:{len(simply_user_rec_list)}")
        longID_simply_user_rec_list = []
        for id in simply_user_rec_list:
            if id[1:len(id)] in answer_id_dict.keys():
                longID_simply_user_rec_list.append(answer_id_dict[id[1:len(id)]])
        if user in commit_log_Dict.keys():
            simply_commit_list = commit_log_Dict[user]
            for recitem_2 in longID_simply_user_rec_list:
                if recitem_2 in simply_commit_list:
                    longID_simply_user_rec_list.remove(recitem_2)
            # print(f"去掉用户{user}longID_simply_user_rec_list:{len(longID_simply_user_rec_list)}")
        popular_rec_dict[user] = longID_simply_user_rec_list
    print(f"流行度推荐字典大小{len(popular_rec_dict)}")
    return popular_rec_dict


def PrintPopular_Rec(n, allday_popular_rec_dict, today_popular_rec_dict, today_count, answer_id_dict, outfile):
    # today_count是当日流行的话题推荐数量，先推荐今日流行 后推荐最近所有天数流行的
    result_file_object = open(outfile, 'w', encoding='UTF-8')
    for user in allday_popular_rec_dict.keys():
        result_file_object.write(user + '\t')
        today_max_popular = today_popular_rec_dict[user]
        allday_max_popular = allday_popular_rec_dict[user]
        index = 0
        for item in today_max_popular:
            answerID = item
            index += 1
            result_file_object.write(answerID + "@" + str(index) + ',')
            if index == n or index == today_count:
                break
        for item in allday_max_popular:
            answerID = item
            index += 1
            result_file_object.write(answerID + "@" + str(index) + ',')
            if index == n:
                break
        result_file_object.write('\n')
    result_file_object.close()
    print("结果输出完成")


if __name__ == '__main__':
    test_set_floder = "E:\\CCIR_online\\test_set"
    allday_hot_file = "E:\\CCIR_online\\popular\\popular_25,26,27,28,29,30.txt"
    today_hot_file = "E:\\CCIR_online\\popular\\popular_30.txt"
    answer_id_dict_file = "E:\\CCIR_online\\20180731\\answer_id_all.dict"
    candidate_answer_file = "E:\\CCIR_online\\20180731\\candidate_online_all.txt"
    commit_log_floder = "E:\\CCIR_online\\commit_log"
    evaryday_test = "E:\\CCIR_online\\20180731\\testing_set_20180730_RecSys_flappyBird.txt"
    outfile = "E:\\CCIR_online\\20180731\\中间文件\\popular_rec.txt"
    testID = get_testID(evaryday_test)
    candidate = loadCandidate(candidate_answer_file)
    UsershowDict = load_UserShowCard(test_set_floder)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    commit_log_Dict = get_commit_log_Dict(commit_log_floder)
    allday_popular_rec_dict = popular_rec(allday_hot_file, answer_id_dict, candidate, UsershowDict, commit_log_Dict, testID)
    today_popular_rec_dict = popular_rec(today_hot_file, answer_id_dict, candidate, UsershowDict, commit_log_Dict, testID)
    PrintPopular_Rec(10, allday_popular_rec_dict, today_popular_rec_dict, 3, answer_id_dict, outfile)





