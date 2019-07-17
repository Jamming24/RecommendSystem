# encoding:utf-8
# 计算思想 筛选出与用户A相似度最高的n个用户，统计这n个用户阅读次数最多的k个答案
# 统计阅读次数，对阅读次数进行归一化处理，作为流行度分数，去流行度为前100的作为用户推荐列表
import os


def loadUserSimilarity(similarity_floder, k):
    # 每个用户取前k个最相关用户    k <= 100
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


def Computer_Popularity_degree(User_Behavior_Dict, User_Similary_dict):
    User_Popularity_dict = dict()
    for userID in User_Behavior_Dict.keys():
        Simple_user_popular_dict = dict()
        if userID in User_Similary_dict.keys():
            related_user_list = User_Similary_dict[userID]
            item_list = User_Behavior_Dict[userID]
            for itemUser in related_user_list:
                if itemUser in User_Behavior_Dict.keys():
                    for relared_item in User_Behavior_Dict[itemUser]:
                        # 相关用户的所阅读过的文章ID
                        if relared_item in Simple_user_popular_dict.keys():
                            count = Simple_user_popular_dict[relared_item]
                            count += 1
                            Simple_user_popular_dict[relared_item] = count
                        else:
                            Simple_user_popular_dict[relared_item] = 1
            for id in item_list:
                if id in Simple_user_popular_dict.keys():
                    del Simple_user_popular_dict[id]
        if len(Simple_user_popular_dict) != 0:
            sort_popular_item = sorted(Simple_user_popular_dict.items(), key=lambda e: e[1], reverse=True)
            User_Popularity_dict[userID] = sort_popular_item
    print(f"推荐用户数量：{len(User_Popularity_dict)}")
    return User_Popularity_dict


def load_answer_dict(answer_id_dict_file):
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        answer_id_dict[line.split('\t')[0]] = line.split('\t')[1][:32]
    file_object.close()
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
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    similarity_floder = "E:\\CCIR\\Similarity_betweenUser_Floder"
    User_Behavior_Dict, test_ID = load_small_test(small_test_file)
    User_Similary_Dict = loadUserSimilarity(similarity_floder, 40)
    User_Popularity_dict = Computer_Popularity_degree(User_Behavior_Dict, User_Similary_Dict)

