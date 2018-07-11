# encoding:utf-8
# 遍历test文件, 对于每个用户看的文章，找到这篇文章最相似的10条文章，最后所有的根据所有最相思的文章
# 取交集为用户进行推荐

import os
import math
import multiprocessing


def loadItem_Similary(similaryFloder):
    Item_Similary_dict = dict()
    file_list = os.listdir(similaryFloder)
    for fileName in file_list:
        filePath = os.path.join(similaryFloder, fileName)
        file_object = open(filePath, 'r', encoding='UTF-8')
        all_line = file_object.readlines()
        for line in all_line:
            similary_dict = []
            t = line.split('\t')
            Item_ID = t[0]
            for index in range(1, len(t)-1):
                tt = t[index].split(':')[1].split(',')
                answer_ID = tt[0][1:len(tt[0])]
                score = float(tt[1][:len(tt[1])-1])
                similary_dict.append([answer_ID, round(score, 8)])
            Item_Similary_dict[Item_ID] = similary_dict
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
                t = answer.split("|")
                start_time = t[1]
                end_time = t[2]
                if int(end_time) != 0:
                    time_difference = int(start_time) - int(end_time)
                    if time_difference > 0:
                        answer_list.append([t[0], 1.0])
                    else:
                        score = 1 / (1 + 1 / (math.exp(time_difference / 60)))
                        answer_list.append([t[0], round(score, 6)])
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


def load_resver_answer_dict(answer_id_dict_file):
    reverse_answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        reverse_answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    file_object.close()
    return reverse_answer_id_dict


def load_candidate(candidate_file, reverse_answer_id_dict):
    candidate_list = []
    file_object = open(candidate_file, 'r', encoding='UTF-8')
    for line in file_object:
        candidate_list.append("A"+reverse_answer_id_dict[line.split('\t')[1][:32]])
    print(len(candidate_list))
    file_object.close()
    return candidate_list


def multiprocessing_computer(part_User_Behavior_Dict, Item_Similary_dict, k):
    # k值表示为与用户喜欢的物品最相关的k个物品
    name = multiprocessing.current_process().name
    print(f"进程{name}开始执行")
    part_User_Recommond_Dict =dict()
    # 使用多进程技术，加快计算速度
    for userID in part_User_Behavior_Dict:
        simiply_user_rec = dict()
        for similaryItem in part_User_Behavior_Dict[userID]:
            # 用户行为 指文章ID 和 得分
            aid = similaryItem[0]
            score = similaryItem[1]
            if aid in Item_Similary_dict.keys():
                similaryList = Item_Similary_dict[aid][:k]
                for everyItem in similaryList:
                    # 待推荐物品
                    itemName = everyItem[0]
                    itemScore = everyItem[1]
                    if itemName in simiply_user_rec:
                        temp_finalScore = simiply_user_rec[itemName]
                        finalScore = temp_finalScore + (score * itemScore)
                        simiply_user_rec[itemName] = finalScore
                    else:
                        simiply_user_rec[itemName] = score * itemScore
        if len(simiply_user_rec) != 0:
            part_User_Recommond_Dict[userID] = simiply_user_rec
    print(f"进程{name}计算结束")
    return part_User_Recommond_Dict


def multiprocessing_manager(User_Behavior_Dict, Item_Similary_dict, n, k):
    # n为进程池最大进程数量
    # k值表示为与用户喜欢的物品最相关的k个物品
    User_Recommond_Dict = dict()
    result_list = []
    pool = multiprocessing.Pool(processes=n)
    temp_User_Behavior_Dict = dict()
    for userID in User_Behavior_Dict.keys():
        if len(temp_User_Behavior_Dict) == 20000:
            part_User_Behavior_Dict = temp_User_Behavior_Dict.copy()
            temp_Item_Similary_dict = Item_Similary_dict.copy()
            result = pool.apply_async(multiprocessing_computer,
                                      args=(part_User_Behavior_Dict, temp_Item_Similary_dict, k))
            result_list.append(result)
            temp_User_Behavior_Dict.clear()
        else:
            temp_User_Behavior_Dict[userID] = User_Behavior_Dict[userID]

    last_part_User_Behavior_Dict = temp_User_Behavior_Dict.copy()
    last__Item_Similary_dict = Item_Similary_dict.copy()
    pool.apply_async(multiprocessing_computer, args=(last_part_User_Behavior_Dict, last__Item_Similary_dict, k))
    temp_User_Behavior_Dict.clear()
    pool.close()
    pool.join()

    for res in result_list:
        User_Recommond_Dict.update(res.get())
    print(f"总用户数量{len(User_Recommond_Dict)}")
    return User_Recommond_Dict


def get_RecommandList(User_Behavior_Dict, User_Recommond_Dict, answer_id_dict, Outfile):
    # k值为与商品最相关的k个商品
    max_relevant_Users_file_object = open(Outfile, 'w', encoding='UTF-8')

    Commit_user_recommond_dict = dict()
    max_relevant_list = []
    for userID in User_Recommond_Dict.keys():
        commit_simiply_user_rec = []
        simiply_user_rec = User_Recommond_Dict[userID]
        # 排序并且取前200个最相关的
        max_relevant = sorted(simiply_user_rec.items(), key=lambda e: e[1], reverse=True)[:200]
        max_relevant_Users_file_object.write(userID+" ")
        for item in max_relevant:
            line = str(item[0])+","+str(item[1])+" "
            max_relevant_list.append(item[0])
            max_relevant_Users_file_object.write(line)
        max_relevant_Users_file_object.write("\n")
        # print(f"去掉重复项之前{len(max_relevant_list)}")
        for li in User_Behavior_Dict[userID]:
            if li[0] in max_relevant_list:
                max_relevant_list.remove(li[0])
        # print(f"去掉重复项之后{len(max_relevant_list)}")
        # print(max_relevant_list)
        for id in max_relevant_list:
            if id[0] == 'A':
                commit_simiply_user_rec.append(answer_id_dict[id[1:len(id)]])
        Commit_user_recommond_dict[userID] = commit_simiply_user_rec
        max_relevant_list.clear()
    max_relevant_Users_file_object.close()
    return Commit_user_recommond_dict


def get_Commit_csv(test_ID, Commit_user_recommond_dict, csv_out_file, n):
    out_file_object = open(csv_out_file, 'w', encoding='UTF-8')
    for user in test_ID:
        if user in Commit_user_recommond_dict.keys():
            result_list = Commit_user_recommond_dict[user]
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
    recommand_list_file = "E:\\CCIR\\ItemCF_Word2Vector_RecommendList.txt"
    Item_Similary_dict = loadItem_Similary(SimilaryFloder)
    User_Behavior_Dict, test_ID = load_small_test(small_test_file)
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    # 7是进程数量 20是与Item最相关的数量
    User_Recommond_Dict = multiprocessing_manager(User_Behavior_Dict, Item_Similary_dict, 7, 20)
    Commit_user_recommond_dict = get_RecommandList(User_Behavior_Dict, User_Recommond_Dict, answer_id_dict, recommand_list_file)
    get_Commit_csv(test_ID, Commit_user_recommond_dict, csv_out_file, 100)
    # multiprocessing_computer(User_Behavior_Dict, Item_Similary_dict, 10)