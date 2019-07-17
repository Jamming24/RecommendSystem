# encoding:utf-8
# 本程序功能 对于完整的测试集合用户，判断用户行为属于哪个话题，然后为用户推荐话题相关的优质答案，
# 判断用户行为属于哪个问题，为用户推荐该问题的其他优质答案
import math


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


def get_User_read_topic(User_Behavior_Dict, answer_id_dict, user_topic_file):
    file_object = open(user_topic_file, 'r', encoding='UTF-8')
    user_topic_dic = dict()
    for line in file_object:
        tt = line.split('\t')
    file_object.close()
    return user_topic_file


if __name__ == '__main__':
    file_object = open("E:\\CCIR\\无标题-2.txt", 'r', encoding='UTF-8')
    file_object2 = open("E:\\CCIR\\无标题-3.txt", 'r', encoding='UTF-8')
    file_object3 = open("E:\\CCIR\\无标题-4.txt", 'w', encoding='UTF-8')

    id = file_object.readlines()
    topic = file_object2.readlines()
    out_list = []
    for it in id:
        print(it)
    # for index in range(len(id)):
    #     out_list.append(id[index]+"\t"+topic[index])
    file_object.close()
    file_object2.close()





