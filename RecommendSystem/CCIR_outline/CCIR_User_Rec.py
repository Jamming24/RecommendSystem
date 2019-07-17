# encoding:utf-8

import os


def loadUserSimilarity(similarity_floder, k):
    # 每个用户取前k个最相关用户
    # assert k <= 100
    User_Similary_Dict = dict()
    filename_list = os.listdir(similarity_floder)
    for file in filename_list:
        all_lines = open(os.path.join(similarity_floder, file), 'r', encoding='UTF-8').readlines()
        for line in all_lines:
            user_list = []
            temp = line.split(":")
            UserID = temp[1][:32]
            if len(temp) >= k+2:
                n = k + 2
            else:
                n = len(temp)
            for i in range(2, n):
                user_list.append(temp[i][1:33])
            if len(user_list) != 0:
                User_Similary_Dict[UserID] = user_list
    return User_Similary_Dict


def loadTest_Behavior(test_set_file):
    User_Behavior_Dict = dict()
    file_object = open(test_set_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        userID = temp[0]
        answer_list = []
        for answer in temp[2].split(","):
            if len(answer) != 0:
                if answer.split("|")[2] != '0':
                    answer_list.append(answer.split("|")[0])
        if len(answer_list) != 0:
            User_Behavior_Dict[userID] = answer_list
    return User_Behavior_Dict


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
    return answer_id_dict


def load_candidate(candidate_file, reverse_answer_id_dict):
    candidate_list = []
    file_object = open(candidate_file, 'r', encoding='UTF-8')
    for line in file_object:
        candidate_list.append("A"+reverse_answer_id_dict[line.split('\t')[1][:32]])
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
    out_file_object.close()


if __name__ == '__main__':
    SimilarityFloder = "E:\\CCIR\\Similarity_betweenUser_Floder"
    test_set_file = "E:\\CCIR\\testing_set_Part.txt"
    csv_out_file = "E:\\CCIR\\result.csv"
    answer_id_dict_file = "E:\\CCIR\\answer_id.dict"
    candidate_file = "E:\\CCIR\\可计算数据\\candidate.txt"
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    User_Similary_Dict = loadUserSimilarity(SimilarityFloder, 15)
    User_Behavior_Dict = loadTest_Behavior(test_set_file)
    candidate_list = load_candidate(candidate_file, answer_id_dict)
    User_Recommond_Dict = get_RecommandList(User_Behavior_Dict, User_Similary_Dict, answer_id_dict, candidate_list)
    test_ID_object = open("E:\\CCIR\\test_ID.txt", 'r', encoding='UTF-8')
    test_ID = []
    for ID in test_ID_object:
        test_ID.append(ID[:32])
    print(len(test_ID))
    get_Commit_csv(test_ID, User_Recommond_Dict, csv_out_file, 100)
