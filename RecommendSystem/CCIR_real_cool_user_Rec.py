# encoding:utf-8
import math


def load_small_testID(small_test_file):
    test_ID = []
    file_object = open(small_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        test_ID.append(temp[0])
    file_object.close()
    print(f"用户ID数量{len(test_ID)}")
    return test_ID


def load_real_cool_user_behavior(real_cool_user_behavior_file):
    # 返回冷启动用户ID
    cool_user_id_list = []
    file_object = open(real_cool_user_behavior_file, 'r', encoding='UTF-8')
    for line in file_object:
        cool_user_id_list.append(line.split('\t')[0])
    file_object.close()
    return cool_user_id_list


def load_real_cool_user_info(all_user_info_file, cool_user_id_list):
    # 返回冷启动用户关注话题
    file_object = open(all_user_info_file, 'r', encoding='UTF-8')
    real_cool_user_topic_dic = dict()
    for line in file_object:
        if line.split('\t')[0] in cool_user_id_list:
            tt = line.split('\t')
            if int(tt[5]) > 0:
                topics = tt[-1].strip('\n').split(",")
                if len(topics) > 0:
                    real_cool_user_topic_dic[tt[0]] = topics
    print(f"冷启动用户信息数量{len(real_cool_user_topic_dic)}")
    return real_cool_user_topic_dic


def load_topic_inverse_answer(topic_inverse_answer_file):
    topic_inverse_answer_dic = dict()
    file_object = open(topic_inverse_answer_file)
    for line in file_object:
        tt = line.split('\t')
        topic_inverse_answer_dic[tt[0]] = tt[1].strip('\n').split(',')[:-1]
    file_object.close()
    return topic_inverse_answer_dic


def estimate_excellent_answer(candidate_answer_file):
    # key为文章ID，值为1的是优质答案 ，值为2的时候是优质+编辑推荐答案，值为0的时候是普通答案
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    candidate_answer_quality = dict()
    for line in file_object:
        tt = line.split('\t')
        # 是否是优质答案
        exe_flag = tt[4]
        # 是否被编辑推荐
        rec_flag = tt[5]
        value = int(exe_flag) + int(rec_flag)
        # 被感谢的次数
        thanks = int(tt[9])
        # 被赞同的次数
        agree = int(tt[10])
        # 被收藏的次数
        save = int(tt[12])
        # 被反对的次数
        disagree = int(tt[13])
        # 收到没有帮助的次数
        nohelp = int(tt[15])
        count = thanks + agree + save - disagree - nohelp
        score = 1 / (1 + 1 / (math.exp(count/1000)))
        # print(tt[0], ">>>>>", score)
        candidate_answer_quality[tt[0]] = value + score
    file_object.close()
    print(f"答案数量{len(candidate_answer_quality)}")
    return candidate_answer_quality


def get_real_cool_user_Rec(real_cool_user_topic_dic, topic_inverse_answer_dic, candidate_answer_quality, outfile):
    # 先得到每个用户关注的话题列表。然后通过topic_inverse_answer找到每个话题对应该的答案列表，
    # 然后再到candidate_answer_quality中得到每个答案的得分，进行排序，前100推荐给用户
    cool_User_Recommond_Dict = dict()
    file_object = open(outfile, 'w', encoding='UTF-8')
    for user in real_cool_user_topic_dic.keys():
        Simply_User_Rec = dict()
        topics = real_cool_user_topic_dic[user]
        for topic in topics:
            if topic in topic_inverse_answer_dic.keys():
                for answer in topic_inverse_answer_dic[topic]:
                    if answer in candidate_answer_quality.keys():
                        if answer in Simply_User_Rec.keys():
                            temp_score = Simply_User_Rec[answer]
                            new_score = temp_score + candidate_answer_quality[answer]
                            Simply_User_Rec[answer] = new_score
                        else:
                            Simply_User_Rec[answer] = candidate_answer_quality[answer]
        max_relevant = sorted(Simply_User_Rec.items(), key=lambda e: e[1], reverse=True)

        cool_User_Recommond_Dict[user] = max_relevant[:200]
    for key in cool_User_Recommond_Dict.keys():
        file_object.write(key+"\t")
        for answer in cool_User_Recommond_Dict[key]:
            file_object.write(answer[0]+":"+str(answer[1])+",")
        file_object.write("\n")
    file_object.close()
    print("输出完成")


def load_120_ItemCF_result(best_result_file):
    file_object = open(best_result_file, 'r', encoding='UTF-8')
    ItemCF_Rec = dict()
    for line in file_object:
        tt = line.split('\t')
        ItemCF_Rec[tt[0]] = tt[1].strip("\n").split(',')
    file_object.close()
    print(f"ItemCF推荐数量:{len(ItemCF_Rec)}")
    return ItemCF_Rec


def load_cool_user_topic_rec(real_cool_user_Reclist_file):
    cool_user_topic_rec_dic = dict()
    file_object = open(real_cool_user_Reclist_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = list()
        tt = line.split('\t')
        for answer in tt[1].strip('\n').split(','):
            temp.append(answer[:32])
        if len(temp) > 1:
            cool_user_topic_rec_dic[tt[0]] = temp[:len(temp)-1]
    file_object.close()
    print(f"冷启动推荐数量{len(cool_user_topic_rec_dic)}")
    return cool_user_topic_rec_dic


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


if __name__ == '__main__':
    # all_user_info_file = "E:\\CCIR\\real_cool_user\\all_user_info.txt"
    # real_cool_user_behavior_file = "E:\\CCIR\\real_cool_user\\testing_set_135089_real_cool_user.txt"
    # topic_inverse_answer_file = "E:\\CCIR\\real_cool_user\\topic_inverse_answer.txt"
    # candidate_answer_file = "E:\\CCIR\\real_cool_user\\candidate_answer.txt"
    # outfile = "E:\\CCIR\\real_cool_user\\real_cool_user_Reclist.txt"
    # # 第一部分 计算中带有得分的用户推荐列表 ################################################
    # cool_user_id_list = load_real_cool_user_behavior(real_cool_user_behavior_file)
    # real_cool_user_topic_dic = load_real_cool_user_info(all_user_info_file, cool_user_id_list)
    # topic_inverse_answer_dic = load_topic_inverse_answer(topic_inverse_answer_file)
    # candidate_answer_quality = estimate_excellent_answer(candidate_answer_file)
    # get_real_cool_user_Rec(real_cool_user_topic_dic, topic_inverse_answer_dic, candidate_answer_quality, outfile)
    # ##################################################################################
    best_result_file = "E:\\CCIR\\ItemCF_best\\120_ItemCf_Rec_result.txt"
    real_cool_user_Reclist_file = "E:\\CCIR\\real_cool_user\\real_cool_user_Reclist.txt"
    csv_out_file = "E:\\CCIR\\real_cool_user\\result.csv"
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    test_ID = load_small_testID(small_test_file)
    ItemCF_Rec = load_120_ItemCF_result(best_result_file)
    cool_user_topic_rec_dic = load_cool_user_topic_rec(real_cool_user_Reclist_file)
    for key in cool_user_topic_rec_dic.keys():
        if key not in ItemCF_Rec.keys():
            ItemCF_Rec[key] = cool_user_topic_rec_dic[key]
    print(len(ItemCF_Rec))
    get_Commit_csv(test_ID, ItemCF_Rec, csv_out_file, 100)