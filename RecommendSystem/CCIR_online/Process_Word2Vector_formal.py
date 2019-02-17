# coding=utf-8
# 把每天的testing_set_20180727_RecSys_flappyBird追加到static_training_set_word2vector_file中进行训练
import os


def load_User_behavior_train_file(behaviorFilePath, static_training_set_word2vector_file):
    count = 0
    Word2Vector_formal_list = []
    file_object = open(behaviorFilePath, 'r', encoding='UTF-8')
    behaviorData = file_object.readlines()
    for data in behaviorData:
        count += 1
        if count == 50000:
            print("50000行处理完成")
            count = 0
        datas = data.split('\t')
        if len(datas) > 2:
            # 用户行为数量
            behavior_num = datas[1]
            if behavior_num != '0' and behavior_num != '1':
                # 用户行为记录
                behavior_content = datas[2]
                answer_questions = behavior_content.split(',')
                behavior_list = []
                for answer_ques in answer_questions:
                    attr = answer_ques.split("|")
                    if len(attr) == 3:
                        action_id = attr[0]
                        if action_id[0] == 'A' and attr[2] != '0':
                            behavior_list.append(action_id)
                Word2Vector_formal_list.append(behavior_list)
    print("固定Word2Vector_formal_list长度：", len(Word2Vector_formal_list))
    file_object.close()
    out_file_object = open(static_training_set_word2vector_file, 'w', encoding='UTF-8')
    for line in Word2Vector_formal_list:
        for item in line:
            out_file_object.write(item+" ")
        out_file_object.write('\n')
    out_file_object.close()


def load_everyday_testingset(everyday_test_file, static_training_set_word2vector_file):
    file_object = open(everyday_test_file, 'r', encoding='UTF-8')
    everyday_Word2Vector = []
    for line in file_object:
        tt = line.split('\t')
        if tt[1] != '0' and tt[1] != '1':
            answers = tt[2]
            simple_line = []
            answers_list = answers.split(',')
            for anw in answers_list:
                attr = anw.split('|')
                if attr[0][0] == 'A' and attr[2] != '0':
                    simple_line.append(attr[0])
            everyday_Word2Vector.append(simple_line)
    print(f"everyday_Word2Vector数量大小:{len(everyday_Word2Vector)}")
    file_object.close()
    out_object = open(static_training_set_word2vector_file, 'a', encoding='UTF-8')
    for everyUser in everyday_Word2Vector:
        for every in everyUser:
            out_object.write(every+" ")
        out_object.write('\n')
    out_object.close()
    print("文件追加成功")


def load_UserReadCard_update(test_set_floder):
    # 加载用户阅读过的内容
    files = os.listdir(test_set_floder)
    UserReadDict = dict()
    for file in files:
        simply_path = os.path.join(test_set_floder, file)
        file_object = open(simply_path, 'r', encoding='UTF-8')
        for line in file_object:
            temp = line.split("\t")
            userID = temp[0]
            if userID in UserReadDict.keys():
                temp_show_answer_dict = UserReadDict[userID]
                for answer in temp[2].split(","):
                    if len(answer) != 0:
                        t = answer.split("|")
                        if int(t[2]) != 0:
                            temp_show_answer_dict[t[0]] = int(t[2])
                UserReadDict[userID] = temp_show_answer_dict
            else:
                show_answer_dict = dict()
                for answer in temp[2].split(","):
                    if len(answer) != 0:
                        t = answer.split("|")
                        if int(t[2]) != 0:
                            show_answer_dict[t[0]] = int(t[2])
                UserReadDict[userID] = show_answer_dict
        file_object.close()
    print(f"UsershowDict大小:{len(UserReadDict)}")
    return UserReadDict


def load_everyday_testingset_update(UserReadDict, static_training_set_word2vector_file):
    out_object = open(static_training_set_word2vector_file, 'a', encoding='UTF-8')
    for userID in UserReadDict.keys():
        simply_answer_dict = UserReadDict[userID]
        if len(simply_answer_dict) > 0:
            sort_answer_tuple = sorted(simply_answer_dict.items(), key=lambda e: e[1], reverse=False)
            for item in sort_answer_tuple:
                out_object.write(item[0] + " ")
            out_object.write('\n')
    out_object.close()
    print("文件追加成功")


if __name__ == '__main__':
    ########################################################
    # behaviorFilePath = "F:\\CCIR_online\\training_test_set.txt"
    # static_training_set_word2vector_file = "F:\\CCIR_online\\static_training_set_word2vector.txt"
    # 处理train训练集合为Word2Vector格式，此后向这个文件追加即可
    # load_User_behavior_train_file(behaviorFilePath, static_training_set_word2vector_file)
    ########################################################
    # 第二部分  追加每天的test数据
    # everyday_test_file = "F:\\CCIR_online\\20180804\\testing_set_20180803_RecSys_flappyBird.txt"
    # last_static_training_set_word2vector_file = "F:\\CCIR_online\\20180804\\中间文件\\static_training_set_word2vector_0803.txt"
    # load_everyday_testingset(everyday_test_file, last_static_training_set_word2vector_file)
    # 第三部分 训练数据处理升级版，讲每个用户的行为轨迹汇总
    test_set_floder = "E:\\CCIR_online\\test_set"
    static_training_set_word2vector_file = "E:\\CCIR_online\\20180809\\中间文件\\static_training_set_word2vector_0808.txt"
    UserReadDict = load_UserReadCard_update(test_set_floder)
    load_everyday_testingset_update(UserReadDict, static_training_set_word2vector_file)