# coding=utf-8

import csv


def staticUserId(file):
    # 判断test中用户ID是否有重复
    file_object = open(file, 'r', encoding='UTF-8')
    files = file_object.readlines()
    count = 0
    user_Id_set = set()

    for line in files:
        user_Id = line.split('\t')[0]
        user_Id_set.add(user_Id)
        count += 1
    # 1351781 行
    print(count, '行')
    # set中含有： 1351781 行
    print('set中含有：', len(user_Id_set), '行')
    file_object.close()


def loadUser_action(FilePath):
    count = 0
    mulitBuyUser = dict()
    User_Behavior = dict()
    flag = False
    csv_file = csv.reader(open(FilePath, 'r'))
    for user_line in csv_file:
        count += 1
        if flag:
            if user_line[0] in User_Behavior.keys():
                simple_user = User_Behavior[user_line[0]]
                # if user_line[1] in simple_userimple_user.keys():
                #     print(user_line)
                simple_user[user_line[1]] = user_line[2] + ',' + user_line[3] + ',' + user_line[4]
                User_Behavior[user_line[0]] = simple_user
            else:
                simple_user = dict()
                simple_user[user_line[1]] = user_line[2] + ',' + user_line[3] + ',' + user_line[4]
                User_Behavior[user_line[0]] = simple_user
        else:
            flag = True
            continue

    print(count)
    return User_Behavior


filePath = 'C:\\Users\\Jamming\\Desktop\\JDATA用户购买时间预测_A榜\\jdata_user_action.csv'
file_path = "E:\\CCIR测试数据\\testing_set.txt"
# be_dic = loadUser_action(filePath)
# print(f'用户数:{len(be_dic.keys())}')
staticUserId(file_path)
