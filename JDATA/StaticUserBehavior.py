# coding=utf-8

import csv


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
be_dic = loadUser_action(filePath)
# count = 0
# for line in be_dic.keys():
#     temp = be_dic[line]
#     for kk in temp.keys():
#         count += 1
# print(f'总量:{count}')
print(f'用户数:{len(be_dic.keys())}')
