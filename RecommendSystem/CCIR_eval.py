# encoding:utf-8


def load_training_set(part_training_test_file, outfile):
    # 取13.5万条训练集合作为测试指标
    User_Behavior_Dict = dict()
    file_object = open(part_training_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        temp = line.split("\t")
        userID = temp[0]
        answer_list = []
        tt = temp[2].split(",")
        for answer in tt[0:len(tt-1)]:
            answer_list.append(answer)
    file_object.close()
    return User_Behavior_Dict

