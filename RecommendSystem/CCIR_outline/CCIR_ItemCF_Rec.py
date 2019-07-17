# encoding:utf-8
# 遍历test文件, 对于每个用户看的文章，找到这篇文章最相似的10条文章，最后所有的根据所有最相思的文章
# 取交集为用户进行推荐

import os


def loadItem_Similary(similaryFloder):
    Item_Similary_dict = dict()
    file_list = os.listdir(similaryFloder)
    for fileName in file_list:
        filePath = os.path.join(similaryFloder, fileName)
        file_object = open(filePath, 'r', encoding='UTF-8')
        all_line = file_object.readlines()
        for line in all_line:
            similary_list = []
            t = line.split('\t')
            Item_ID = t[0]
            for index in range(1, len(t)-1):
                similary_list.append(t[index].split('(')[1].split(',')[0])
            Item_Similary_dict[Item_ID] = similary_list
        file_object.close()
    print(f"相似度字典大小{len(Item_Similary_dict)}")
    return Item_Similary_dict


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


def load_answer_dict(answer_id_dict_file):
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        answer_id_dict[line.split('\t')[0]] = line.split('\t')[1][:32]
    file_object.close()
    return answer_id_dict


def get_RecommandList(User_Behavior_Dict, Item_Similary_dict, answer_id_dict, k):
    # k值为与商品最相关的k个商品
    User_Recommond_Dict = dict()
    temp = []
    for userID in User_Behavior_Dict.keys():
        simiply_user_rec = []
        for similaryItem in User_Behavior_Dict[userID]:
            if similaryItem in Item_Similary_dict.keys():
                temp = list(set(temp).union(set(Item_Similary_dict[similaryItem][:k])))
        if userID in User_Behavior_Dict.keys():
            temp = list(set(temp).difference(set(User_Behavior_Dict[userID])))
            for a in temp:
                if a[0] == 'A':
                    simiply_user_rec.append(answer_id_dict[a[1:len(a)]])
        temp.clear()

        if len(simiply_user_rec) != 0:
            User_Recommond_Dict[userID] = simiply_user_rec
            # print(userID)
            # print(f"{userID}推荐数量:{len(simiply_user_rec)}")
    return User_Recommond_Dict


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
    SimilaryFloder = "E:\\CCIR\\Item_Similary"
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    csv_out_file = "E:\\CCIR\\result.csv"
    answer_id_dict_file = "E:\\CCIR\\answer_id.dict"
    candidate_file = "E:\\CCIR\\candidate.txt"
    Item_Similary_dict = loadItem_Similary(SimilaryFloder)
    User_Behavior_Dict, test_ID = load_small_test(small_test_file)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    User_Recommond_Dict = get_RecommandList(User_Behavior_Dict, Item_Similary_dict, answer_id_dict, 15)
    get_Commit_csv(test_ID, User_Recommond_Dict, csv_out_file, 100)

