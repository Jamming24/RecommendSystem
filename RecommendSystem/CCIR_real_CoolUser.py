# encoding:utf-8


def data_pretreatment():
    result_file = open("E:\\CCIR\\80-result\\result.csv", 'r', encoding='UTF-8')
    small_test_file = open("E:\\CCIR\\testing_set_135089.txt", 'r', encoding='UTF-8')
    real_cool_user = open("E:\\CCIR\\testing_set_135089_real_cool_user.txt", 'w', encoding='UTF-8')
    count = 0
    cool_list = []
    for line in result_file:
        count += 1
        temp = line.split(',')
        if temp[0] == '-1':
            cool_list.append(count)
    test_dic = dict()
    linenum = 0
    for line in small_test_file:
        linenum += 1
        test_dic[linenum] = line
    for li in cool_list:
        real_cool_user.write(test_dic[li])

    result_file.close()
    small_test_file.close()
    real_cool_user.close()
    print(f"80-result包含未推荐用户：{len(cool_list)}")
    print(f"字典大小{len(test_dic)}")


def load_real_cool_user(real_cool_user_file):
    real_cool_user_behavior_dic = dict()
    real_cool_user_list = []
    real_cool_user_object = open(real_cool_user_file, 'r', encoding='UTF-8')
    for line in real_cool_user_object:
        tt = line.split("\t")
        userID = tt[0]
        real_cool_user_list.append(userID)
        if tt[1] != '0':
            real_cool_user_behavior_dic[userID] = tt[2]
    real_cool_user_object.close()
    return real_cool_user_behavior_dic, real_cool_user_list


def load_all_user_info(all_user_info_file):
    User_focus_topic = dict()
    file_object = open(all_user_info_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        if tt[5] != '0':
            User_focus_topic[tt[0]] = tt[26]
    file_object.close()
    print(f"有关注话题的用户数量：{len(User_focus_topic)}")
    return User_focus_topic


def load_question_info(question_info_file,Out_file):
    topic_inverse_question = dict()
    file_object = open(question_info_file, 'r', encoding='UTF-8')
    Out_file_object = open(Out_file, 'w', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        questionID = tt[0]
        topic_list = tt[7].split(',')
        for index in range(0, len(topic_list)-1):
            if topic_list[index] in topic_inverse_question.keys():
                question_list = topic_inverse_question[topic_list[index]]
                question_list.append(questionID)
                topic_inverse_question[topic_list[index]] = question_list
            else:
                topic_inverse_question[topic_list[index]] = [questionID]
        last_topic = topic_list[len(topic_list)-1][:32]
        if last_topic in topic_inverse_question.keys():
            question_list = topic_inverse_question[last_topic]
            question_list.append(questionID)
            topic_inverse_question[last_topic] = question_list
        else:
            topic_inverse_question[last_topic] = [questionID]
    print(f"话题数量{len(topic_inverse_question)}")
    for key in topic_inverse_question.keys():
        # key 表示话题ID
        Out_file_object.write(key+"\t")
        for questionID in topic_inverse_question[key]:
            # questionID表示问题ID
            Out_file_object.write(questionID+",")
        Out_file_object.write("\n")
    print(f"话题问题转换表写出完成")
    file_object.close()
    Out_file_object.close()


def load_answer_info(answer_info_file,Out_file):
    topic_inverse_question = dict()
    file_object = open(answer_info_file, 'r', encoding='UTF-8')
    Out_file_object = open(Out_file, 'w', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        questionID = tt[0]
        topic_list = tt[7].split(',')
        for index in range(0, len(topic_list) - 1):
            if topic_list[index] in topic_inverse_question.keys():
                question_list = topic_inverse_question[topic_list[index]]
                question_list.append(questionID)
                topic_inverse_question[topic_list[index]] = question_list
            else:
                topic_inverse_question[topic_list[index]] = [questionID]
        last_topic = topic_list[len(topic_list) - 1][:32]
        if last_topic in topic_inverse_question.keys():
            question_list = topic_inverse_question[last_topic]
            question_list.append(questionID)
            topic_inverse_question[last_topic] = question_list
        else:
            topic_inverse_question[last_topic] = [questionID]
    print(f"话题数量{len(topic_inverse_question)}")
    for key in topic_inverse_question.keys():
        # key 表示话题ID
        Out_file_object.write(key + "\t")
        for questionID in topic_inverse_question[key]:
            # questionID表示问题ID
            Out_file_object.write(questionID + ",")
        Out_file_object.write("\n")
    print(f"话题问题转换表写出完成")
    file_object.close()
    Out_file_object.close()


def get_RecommandList(User_Behavior_Dict, User_Similary_Dict, answer_id_dict, candidate_list):
    User_Recommond_Dict = dict()
    temp = []
    for userID in User_Similary_Dict.keys():
        simiply_user_rec = []
        for similaryUser in User_Similary_Dict[userID]:
            if similaryUser in User_Behavior_Dict.keys():
                temp = list(set(temp).union(set(User_Behavior_Dict[similaryUser])))
        if userID in User_Behavior_Dict.keys():
            temp = list(set(temp).difference(set(User_Behavior_Dict[userID])))
            temp = list(set(temp).intersection(set(candidate_list)))
            for a in temp:
                if a[0] == 'A':
                    simiply_user_rec.append(answer_id_dict[a[1:len(a)]])
        temp.clear()

        if len(simiply_user_rec) != 0:
            User_Recommond_Dict[userID] = simiply_user_rec
    return User_Recommond_Dict


def load_answer_dict(answer_id_dict_file):
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        answer_id_dict[line.split('\t')[0]] = line.split('\t')[1][:32]
    file_object.close()
    return answer_id_dict


def get_all_answerID(taining_test_file, answer_id_dict_file):
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    file_object = open(taining_test_file, 'r', encoding='UTF-8')
    answer_list = set()
    for line in file_object:
        tt = line.split('\t')
        answers = tt[2].split(',')
        for an in answers:
            answerID = an.split("|")[0]
            if len(answerID) > 0 and answerID[0] == 'A':
                answer_list.add(answer_id_dict[answerID[1:len(answerID)]])
    print(f"不重复答案ID数量：{len(answer_list)}")
    return answer_list


def get_revelent_answer(answer_list, answer_info_file, new_answer_info_file):
    answer_dic = dict()
    answer_file = open(answer_info_file, 'r', encoding='UTF-8')
    new_answer_file = open(new_answer_info_file, 'w', encoding='UTF-8')
    for line in answer_file:
        answer_dic[line.split('\t')[0]] = line
    print(f"总答案数量{len(answer_dic)}")
    for an in answer_list:
        if an in answer_dic.keys():
            new_answer_file.write(answer_dic[an])
    answer_file.close()
    new_answer_file.close()
    print("写出完成")


if __name__ == '__main__':
    real_cool_user_file = "E:\\CCIR\\testing_set_135089_real_cool_user.txt"
    all_user_info_file = "E:\\CCIR\\all_user_info.txt"
    question_info_file = "E:\\CCIR\\question_infos.txt"
    Out_file = "E:\\CCIR\\topic_inverse_question.txt"
    taining_test_file = "E:\\CCIR\\training_set_Part.txt"
    answer_id_dict_file = "E:\\CCIR\\answer_id.dict"
    answer_info_file = "E:\\CCIR\\answer_infos_part.txt"
    new_answer_info_file = "E:\\CCIR\\new_answer_infos.txt"
    answer_list = get_all_answerID(taining_test_file, answer_id_dict_file)
    get_revelent_answer(answer_list, answer_info_file, new_answer_info_file)
    # real_cool_user_behavior_dic, real_cool_user_list = load_real_cool_user(real_cool_user_file)
    # User_focus_topic = load_all_user_info(all_user_info_file)
    # load_question_info(question_info_file, Out_file)
    # temp_set = set(real_cool_user_list).intersection(set(User_focus_topic.keys()))
    # print(f"有话题冷启动用户{len(temp_set)}")
    # print(temp_set)


