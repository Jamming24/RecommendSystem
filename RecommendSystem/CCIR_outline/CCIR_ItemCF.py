# encoding:utf-8
# 直接计算与test中的问题Id最相关并且在候选集合中的答案ID 可以减少计算量
# 首先 加载test中的所有问题ID 计算与每个Id最相关的100问题（并且只保留在候选集合中的） 返回一个集合
# 遍历test 对于每条用户行为 按照时间从高到排序，最近发生的放到前面，取与每个行为最相关的100条，存到字典中
# 作为推荐结果
#
import multiprocessing
import numpy.linalg as li
import numpy as np
import os


def loadTestAnswer(testfilePath):
    test_User_behavior_dic = dict()
    behavior_ID_list = []
    file_object = open(testfilePath, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split('\t')
        userID = t[0]
        answer_list = t[2].split(",")
        for answer in answer_list:
            behavior_ID_list.append(answer.split("|")[0])
        test_User_behavior_dic[userID] = answer_list
    behavior_ID_list = list(set(behavior_ID_list))
    print(f"用户数量：{len(test_User_behavior_dic)}")
    print(f"答案数量：{len(behavior_ID_list)}")
    return test_User_behavior_dic, behavior_ID_list


def multiprocessing_computer(behavior_ID_list, Word2Vector_dic, candidate, out_Floder):
    # 用户行为记录字典, 用户行为记录倒排表字典, 保存前n个最相关用户
    # 使用进程池技术
    pool = multiprocessing.Pool(8)
    user_list_temp = []
    file_count = 1
    for user in behavior_ID_list:
        user_list_temp.append(user)
        if len(user_list_temp) == 500:
            part_user_list = user_list_temp.copy()
            out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
            temp_Word2Vector_dic = Word2Vector_dic.copy()
            temp_candidate = candidate.copy()
            pool.apply_async(cosValue, args=(part_user_list, temp_Word2Vector_dic, temp_candidate, out_file))
            # cosValue(part_user_list, temp_Word2Vector_dic, temp_candidate, out_file)
            user_list_temp.clear()
            file_count += 1

    print('最后一部分的用户数：', len(user_list_temp))
    out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
    pool.apply_async(cosValue, args=(user_list_temp, Word2Vector_dic, candidate, out_file))
    pool.close()
    pool.join()
    print(f'实际用户数：{len(behavior_ID_list)}')


def loadCandidate(candidate_answer_file, answer_id_dict):
    candidate = []
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    for line in file_object:
        long_ID = line.split('\t')[0]
        if long_ID in answer_id_dict.keys():
            # print(answer_id_dict[long_ID])
            candidate.append("A"+answer_id_dict[long_ID])
    print(f"候选集数量：{len(candidate)}")
    return candidate


def load_answer_dict(answer_id_dict_file):
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    return answer_id_dict


def loadVector(Word2VectorFile):
    Word2Vector_dic = dict()
    temp = []
    file_object = open(Word2VectorFile, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split(" ")
        answer = t[0]
        for num in t[1:len(t)]:
            temp.append(float(num))
        vector = np.array(temp)
        temp.clear()
        Word2Vector_dic[answer] = vector
    print(f"词向量个数为:{len(Word2Vector_dic)}")
    return Word2Vector_dic


def cosValue(behavior_ID_list, Word2Vector_dic, candidate, outfile):
    print(f'线程{outfile}计算开始')
    Item_similary_dic = dict()
    file_object = open(outfile, 'w', encoding='UTF-8')
    for aid in behavior_ID_list:
        aid_list = []
        for ca in candidate:
            if aid in Word2Vector_dic.keys() and ca in Word2Vector_dic.keys():
                num = np.dot(Word2Vector_dic[aid], Word2Vector_dic[ca])
                denom = li.norm(Word2Vector_dic[aid]) * li.norm(Word2Vector_dic[ca])
                cos = num / denom
                aid_list.append((ca, cos))
        if len(aid_list) != 0:
            Item_similary_dic[aid] = sorted(aid_list)[:100]
    for key in Item_similary_dic.keys():
        file_object.write(key+"\t")
        for item in Item_similary_dic[key]:
            file_object.write(str(item)+"\t")
        file_object.write("\n")
    file_object.close()
    print(f'线程{outfile}计算完成')


if __name__ == '__main__':
    testfilePath = "E:\\CCIR\\testing_set_Part.txt"
    candidate_answer_file = "E:\\CCIR\\candidate_answer.txt"
    Word2VectorFile = "E:\\CCIR\\word2Vector.txt"
    answer_id_dict_file = "E:\\CCIR\\answer_id.dict"
    Item_similary_Floder = "E:\\CCIR\\Item_Similary"
    answer_id_dict = load_answer_dict(answer_id_dict_file)
    candidate = loadCandidate(candidate_answer_file, answer_id_dict)
    test_User_behavior_dic, behavior_ID_list = loadTestAnswer(testfilePath)
    Word2Vector_dic = loadVector(Word2VectorFile)
    # cosValue(behavior_ID_list, Word2Vector_dic, candidate, Item_similary_file)
    multiprocessing_computer(behavior_ID_list, Word2Vector_dic, candidate, Item_similary_Floder)


