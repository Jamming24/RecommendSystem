# encoding:utf-8


def load_User_behavior_inverse_table(behaviorFilePath):
    # 用户行为记录倒排表字典
    user_behavior_inverse_table = dict()
    file_object = open(behaviorFilePath, 'r', encoding='UTF-8')
    behaviorData = file_object.readlines()
    for data in behaviorData:
        behavior_list = list()
        datas = data.split('\t')
        # 用户ID
        user_Id = datas[0]
        # 用户行为数量
        # behavior_num = datas[1]
        # 用户行为记录
        behavior_content = datas[2]
        # 建立用户行为倒排表
        answer_questions = behavior_content.split(',')
        for answer_ques in answer_questions:
            attr = answer_ques.split("|")
            if len(attr) == 3:
                action_id = attr[0]
                end_time = attr[2]
                if int(end_time) != 0:
                    behavior_list.append(action_id)
                    if action_id in user_behavior_inverse_table.keys():
                        user_dict = user_behavior_inverse_table[action_id]
                        user_dict[user_Id] = end_time
                        user_behavior_inverse_table[action_id] = user_dict
                    else:
                        user_dict = dict()
                        user_dict[user_Id] = end_time
                        user_behavior_inverse_table[action_id] = user_dict
                # 讲用户行为存到用户行为字典
    print("user_behavior_inverse_table长度：", len(user_behavior_inverse_table))
    file_object.close()
    return user_behavior_inverse_table


def Print_user_behavior_inverse_table(user_behavior_inverse_table, inverse_table_file, Word2Vectorfile):
    inverse_table_file_object = open(inverse_table_file, 'w', encoding='UTF-8')
    Word2Vectorfile_object = open(Word2Vectorfile, 'w', encoding='UTF-8')
    for aid in user_behavior_inverse_table:
        inverse_table_file_object.write(aid+" ")
        Word2Vectorfile_object.write(aid+" ")
        sort_UserList = sorted(user_behavior_inverse_table[aid].items(), key=lambda e: e[1], reverse=True)
        for userItem in sort_UserList:
            inverse_table_file_object.write(userItem[0]+","+userItem[1]+" ")
            Word2Vectorfile_object.write(userItem[0]+" ")
        inverse_table_file_object.write("\n")
        Word2Vectorfile_object.write("\n")
    inverse_table_file_object.close()
    Word2Vectorfile_object.close()
    print("inverse_table_file写出完成")
    print("Word2Vectorfile写出完成")


if __name__ == '__main__':
    behaviorFilePath = "E:\\CCIR\\testing_set_Part.txt"
    inverse_table_file = "E:\\CCIR\\test_user_inverse_table.txt"
    Word2Vectorfile ="E:\\CCIR\\test_Word2Vectorfile.txt"
    user_behavior_inverse_table = load_User_behavior_inverse_table(behaviorFilePath)
    Print_user_behavior_inverse_table(user_behavior_inverse_table, inverse_table_file, Word2Vectorfile)

