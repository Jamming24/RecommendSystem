# encoding:utf-8
import math


def load_UserCF_Similary(User_Similary_file, k):
    # 加载用户相似度得分文件 k值为最相关的k个用户
    related_User_list = set()
    User_Similary_dict = dict()
    file_object = open(User_Similary_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split(" ")
        userID = tt[0]
        Similary_User = tt[1:k-1]
        Similary_UserScore_list = []
        for item in Similary_User:
            t = item.split(",")
            Similary_UserScore_list.append([t[0], t[1]])
            related_User_list.add(t[0])
        User_Similary_dict[userID] = Similary_UserScore_list
    file_object.close()
    print(f"用户相似度大小：{len(User_Similary_dict)}")
    print(f"相关用户数量{len(related_User_list)}")
    return User_Similary_dict, related_User_list


def load_test_train_behavior(training_test_set_file, related_User_list):
    related_User_Behavior_Dict = dict()
    file_object = open(training_test_set_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.strip("\n").split("\t")
        userID = temp[0]
        if userID in related_User_list:
            answer_list = []
            for answer in temp[2].split(","):
                if len(answer) != 0:
                    t = answer.split("|")
                    start_time = t[1]
                    end_time = t[2]
                    if int(end_time) != 0:
                        time_difference = int(start_time) - int(end_time)
                        if time_difference >= 0 or int(start_time) == 0:
                            answer_list.append([t[0], 1.0])
                        else:
                            # print(answer)
                            score = 1 / (1 + 1 / (math.exp(time_difference / 60)))
                            answer_list.append([t[0], round(score, 6)])
            if len(answer_list) != 0:
                related_User_Behavior_Dict[userID] = answer_list
    print(f"总用户数量{len(User_Behavior_Dict)}")
    file_object.close()
    return related_User_Behavior_Dict


def load_small_test(small_test_file):
    test_ID = []
    User_Behavior_Dict = dict()
    file_object = open(small_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        userID = temp[0]
        test_ID.append(userID)
        answer_list = []
        for answer in temp[2].split(","):
            if len(answer) != 0:
                if answer.split("|")[2] != '0':
                    answer_list.append(answer.split("|")[0])
        if len(answer_list) != 0:
            User_Behavior_Dict[userID] = answer_list
    print(f"test用户数量{len(User_Behavior_Dict)}")
    file_object.close()
    print(f"用户ID数量{len(test_ID)}")
    return User_Behavior_Dict, test_ID


def computer_Rec_list(User_Behavior_Dict, candidate_list, User_Similary_dict, related_User_Behavior_Dict, answer_id_dict, score_out_flie):
    # 保留前n项推荐结果
    # 得到用户推荐列表字典
    User_Recommond_Dict = dict()
    out_file = open(score_out_flie, 'w', encoding='UTF-8')
    temp_list = set()
    for userID in User_Behavior_Dict.keys():
        Simple_user_rec_dict = dict()
        if userID in User_Similary_dict.keys():
            related_user_list = User_Similary_dict[userID]
            delet_list = User_Behavior_Dict[userID]
            for item in related_user_list:
                related_userID = item[0]
                related_score = float(item[1])
                if related_userID in related_User_Behavior_Dict.keys():
                    for relared_item in related_User_Behavior_Dict[related_userID]:
                        temp_list.add(relared_item[0])
                        behavior_score = float(relared_item[1])
                        new_score = related_score * behavior_score
                        Simple_user_rec_dict[relared_item[0]] = new_score
                        # print(related_score, ">>>>", behavior_score, ">>>>", new_score)
            temp_list = temp_list.difference(set(delet_list))
            temp_list.intersection(candidate_list)
            temp_dict = dict()
            for key in Simple_user_rec_dict.keys():
                if key in temp_list:
                    temp_dict[key] = Simple_user_rec_dict[key]
            temp_list.clear()
            del delet_list
            del Simple_user_rec_dict
            Simple_user_rec_tuple_list = sorted(temp_dict.items(), key=lambda e: e[1], reverse=True)
            temp_dict.clear()
            if len(Simple_user_rec_tuple_list) != 0:
                commit_simiply_user_rec = []
                for id in Simple_user_rec_tuple_list:
                    if id[0][0] == 'A':
                        out_file.write(userID+" ")
                        out_file.write(id[0]+":"+str(id[1]))
                        commit_simiply_user_rec.append(answer_id_dict[id[0][1:len(id[0])]])
                User_Recommond_Dict[userID] = commit_simiply_user_rec
                out_file.write("\n")
    print(f"推荐用户数量：{len(User_Recommond_Dict)}")
    out_file.close()
    return User_Recommond_Dict

# def multiprocessing_manager():


def load_answer_dict(answer_id_dict_file):
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        answer_id_dict[line.split('\t')[0]] = line.split('\t')[1][:32]
    return answer_id_dict


def load_resver_answer_dict(answer_id_dict_file):
    reverse_answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        reverse_answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    file_object.close()
    return reverse_answer_id_dict


def load_candidate(candidate_file, answer_id_dict_file):
    reverse_answer_id_dict = load_resver_answer_dict(answer_id_dict_file)
    candidate_list = set()
    file_object = open(candidate_file, 'r', encoding='UTF-8')
    for line in file_object:
        candidate_list.add("A"+reverse_answer_id_dict[line.split('\t')[1][:32]])
    print(len(candidate_list))
    return candidate_list


def get_Commit_csv(test_ID, User_Recommond_Dict, csv_out_file, n):
    out_file_object = open(csv_out_file, 'w', encoding='UTF-8')
    for user in test_ID:
        if user in User_Recommond_Dict.keys():
            result_list = User_Recommond_Dict[user]
            for index in range(0, n):
                if index < len(result_list):
                    ansID = result_list[index]
                    commit_ID = ansID[:4] + ansID[len(ansID)-4:len(ansID)]
                    out_file_object.write(commit_ID + ",")
                else:
                    out_file_object.write("-1,")
        else:
            for i in range(0, n):
                if i == 99:
                    out_file_object.write("-1")
                else:
                    out_file_object.write("-1,")
        out_file_object.write("\n")
    print("result.csv输出完成")
    out_file_object.close()


if __name__ == '__main__':
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    User_Similary_file = "E:\\CCIR\\part_UserCF_UserSimilary_baseOnWord2Vector.txt"
    training_test_set_file = "E:\\CCIR\\part_training_test.txt"
    candidate_file = "E:\\CCIR\\candidate.txt"
    answer_id_dict_file = "E:\\CCIR\\answer_id.dict"
    score_out_flie = "E:\\CCIR\\UserCF_Word2Vector_RecScore"
    csv_out_file = "E:\\CCIR\\result.csv"
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    User_Behavior_Dict, userlist = load_small_test(small_test_file)
    User_Similary_dict, related_User_list = load_UserCF_Similary(User_Similary_file, 20)
    related_User_Behavior_Dict = load_test_train_behavior(training_test_set_file, related_User_list)
    candidate_list = load_candidate(candidate_file, answer_id_dict_file)
    User_Recommond_Dict = computer_Rec_list(User_Behavior_Dict, candidate_list, User_Similary_dict, related_User_Behavior_Dict, answer_id_dict, score_out_flie)

    get_Commit_csv(userlist, User_Recommond_Dict, csv_out_file, 100)







