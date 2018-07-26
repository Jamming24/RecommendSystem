# encoding:utf-8
# 最后删除  为用户推荐 用户缺没有阅读的文章


def load_small_testID(small_test_file):
    test_ID = []
    file_object = open(small_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        test_ID.append(temp[0])
    file_object.close()
    print(f"用户ID数量{len(test_ID)}")
    return test_ID


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
            # print(user)
            for i in range(0, n):
                if i == 99:
                    out_file_object.write("-1")
                else:
                    out_file_object.write("-1,")
        out_file_object.write("\n")
    out_file_object.close()


def get_ItemCF():
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    ItemCF_csv = open("E:\\CCIR\\Combine_final\\ItemCF_result_with_question.csv", 'r', encoding='UTF-8')
    test_ID = load_small_testID(small_test_file)
    ItemCF_dic = dict()
    index = 0
    for item in ItemCF_csv:
        tt = item.strip('\n').split(',')
        simply_item_dic = dict()
        max_score = len(tt) - 1
        for i in range(0, len(tt) - 1):
            if tt[i] == '-1':
                break
            else:
                simply_item_dic[tt[i]] = max_score
                max_score -= 1
        ItemCF_dic[test_ID[index]] = simply_item_dic
        index += 1
    ItemCF_csv.close()
    print(f"ItemCF数量:{len(ItemCF_dic)}")
    return ItemCF_dic


def get_UserCF():
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    UserCF_csv = open("E:\\CCIR\\Combine_final\\UserCF_6.4result.csv", 'r', encoding='UTF-8')
    test_ID = load_small_testID(small_test_file)
    UserCF_dic = dict()
    index = 0
    for user in UserCF_csv:
        tt = user.strip('\n').split(',')
        simply_user_list = []
        for i in range(0, len(tt) - 1):
            if tt[i] == '-1':
                break
            else:
                simply_user_list.append(tt[i])
        UserCF_dic[test_ID[index]] = simply_user_list
        index += 1
    UserCF_csv.close()
    print(f"UserCF_dic数量:{len(UserCF_dic)}")
    return UserCF_dic


if __name__ == '__main__':
    # baseOnQuestion_csv = open("E:\\CCIR\\Combine_final\\CCIR_baseOnQuestion_result.csv", 'r', encoding='UTF-8')
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    csv_out_file = "E:\\result.csv"
    test_ID = load_small_testID(small_test_file)
    ItemCF_dic = get_ItemCF()
    UserCF_dic = get_UserCF()
    Combine_dic = dict()
    for item in ItemCF_dic.keys():
        simply_dic = dict()
        if item in UserCF_dic.keys():
            temp_dict = ItemCF_dic[item]
            # print(temp_dict)
            temp_list = UserCF_dic[item]
            # print(temp_list)
            for recitem in temp_dict.keys():
                if recitem in temp_list:
                    score = temp_dict[recitem]
                    score += 0.5
                    simply_dic[recitem] = score
                else:
                    simply_dic[recitem] = temp_dict[recitem]
            del temp_dict
        else:
            simply_dic = ItemCF_dic[item].copy()
        new_sored = sorted(simply_dic.items(), key=lambda e: e[1], reverse=True)
        new_Simply_list = []
        for newitem in new_sored:
            new_Simply_list.append(newitem[0])
        Combine_dic[item] = new_Simply_list
    get_Commit_csv(test_ID, Combine_dic, csv_out_file, 100)



