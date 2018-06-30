# encoding:utf-8


def loadTrainSet(train_file, answer_id_Dict):
    Train_Set_dic = dict()
    file_object = open(train_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split('\t')
        User_ID = temp[0]
        answer_ID = temp[len(temp)-1]
        answer_time = temp[len(temp)-3]
        Behavior_list = temp[2].split(",")
        Behavior_list.append("A"+answer_id_Dict[answer_ID]+"|0"+"|"+answer_time[:10])
        if "" in Behavior_list:
            Behavior_list.remove("")
        if User_ID in Train_Set_dic.keys():
            temp_list = Train_Set_dic[User_ID]
            # temp_list.extend(Behavior_list)
            temp_list = list(set(temp_list).union(set(Behavior_list)))
            Train_Set_dic[User_ID] = temp_list
        else:
            Train_Set_dic[User_ID] = Behavior_list
    file_object.close()
    return Train_Set_dic


def load_answer_id_Dict(dictfile):
    answer_id_Dict = dict()
    file_object = open(dictfile)
    for line in file_object:
        t = line.split("\t")
        answer_id_Dict[t[1]] = t[0]
    return answer_id_Dict


def DataDenoising(Train_Set_dic, OutPath):
    Clean_Train_Set = dict()
    for User in Train_Set_dic.keys():
        temp = dict()
        Behavior_list = Train_Set_dic[User]
        for answer in Behavior_list:
            time = answer.split("|")
            if time[2] != '0':
                temp[time[2]] = answer
        temp_list = sorted(temp.keys())
        Clean_list = []
        for index in range(0, len(temp_list)):
            Clean_list.append(temp[temp_list[index]])
        temp.clear()
        temp_list.clear()
        Clean_Train_Set[User] = Clean_list
    file_object = open(OutPath, 'w', encoding='UTF-8')
    for user in Clean_Train_Set:
        list = Clean_Train_Set[user]
        line = user+"\t"+str(len(list))+"\t"
        for li in list:
            line = line + li + ","
        file_object.write(line+'\n')
    file_object.close()


def get_IDfocusOnlessThree(User_info_file, cool_user_file):
    user_list = []
    cool_ID = open(cool_user_file, 'w', encoding='UTF-8')
    file_object = open(User_info_file, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split("\t")
        User = t[0]
        num = t[5]
        # 关注话题数  其中「用户关注话题量在 3 以下（包括 3）为冷启动用户
        if int(num) <= 3:
            user_list.append(User)
    print(f"关注数量少于等于3的用户数:{len(user_list)}")
    for id in user_list:
        cool_ID.write(id+"\n")
    cool_ID.close()
    return user_list


def loadTest_Set(test_file, lessFocusUserId):
    # 且样本中行为数（交互 + 搜索词）小于 20 的样本」为冷启动用户样本
    Cool_User_list = []
    file = open(test_file, 'r', encoding='UTF-8')
    for line in file:
        t = line.split('\t')
        user = t[0]
        benum = t[1]
        querynum = t[len(t)-2]
        if (int(benum)+int(querynum)) < 20:
            Cool_User_list.append(user)
    print(f"行为查询数量之和小于20的用户数:{len(Cool_User_list)}")
    Cool_User_list = list(set(Cool_User_list).intersection(set(lessFocusUserId)))
    print(f"冷启动用户数量：{len(Cool_User_list)}")
    return Cool_User_list


def getCool_user_test(testfile, cool_user, Outfile):
    cool_lines = []
    file = open(testfile, 'r', encoding='UTF-8')
    OutFile_object = open(Outfile, 'w', encoding='UTF-8')
    for line in file:
        user = line.split('\t')[0]
        if user in cool_user:
            cool_lines.append(line)
    print(f"冷启动用户总数:{len(cool_lines)}")
    for data in cool_lines:
        OutFile_object.write(data)
    file.close()
    OutFile_object.close()


if __name__ == '__main__':
    train_file = "E:\\CCIR\\training_set_Part.txt"
    dictfile = "E:\\CCIR\\answer_id.dict"
    OutPath = "E:\\CCIR\\purity_training_set.txt"
    user_info_file = "E:\\CCIR\\user_info_Part.txt"
    cool_user_file = "E:\\CCIR\\focusOnlessThressUserID.txt"
    test_file = "E:\\CCIR\\testing_set_Part.txt"
    cool_user_test = "E:\\CCIR\\cool_user_testing_set_Part.txt"
    # answer_id_Dict = load_answer_id_Dict(dictfile)
    # Train_Set = loadTrainSet(train_file, answer_id_Dict)
    # DataDenoising(Train_Set, OutPath)
    lessfocus_list = get_IDfocusOnlessThree(user_info_file, cool_user_file)
    cool_user_ID = loadTest_Set(test_file, lessfocus_list)
    getCool_user_test(test_file, cool_user_ID, cool_user_test)
    # a = [2, 3, 4, 5]
    # b = [2, 5, 8]
    # print(list(set(a).union(set(b))))
