# coding=utf-8

import numpy.linalg as li
import numpy as np
import multiprocessing
import os


def load_every_test_read_answer(every_testfilePath):
    behavior_ID_set = set()
    file_object = open(every_testfilePath, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split('\t')
        answer_list = t[2].split(",")
        for answer in answer_list:
            item = answer.split("|")
            if len(item) == 3 and item[2] != '0':
                behavior_ID_set.add(item[0])
    print(f"答案数量：{len(behavior_ID_set)}")
    return behavior_ID_set


def multiprocessing_computer(n, behavior_ID_list, Word2Vector_dic, candidate_shortID_set, out_Floder):
    # 用户行为记录字典, 用户行为记录倒排表字典, 保存前n个最相关用户
    # 使用进程池技术
    pool = multiprocessing.Pool(n)
    print(f"candidate_shortID_set数量:{len(candidate_shortID_set)}")
    user_list_temp = []
    file_count = 1
    for user in behavior_ID_list:
        user_list_temp.append(user)
        if len(user_list_temp) == 5000:
            part_user_list = user_list_temp.copy()
            out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
            pool.apply_async(cosValue, args=(part_user_list, Word2Vector_dic, candidate_shortID_set, out_file))
            # cosValue(part_user_list, Word2Vector_dic, candidate_shortID_set, out_file)
            user_list_temp.clear()
            file_count += 1

    print('最后一部分的用户数：', len(user_list_temp))
    out_file = os.path.join(out_Floder, f"Similarity_betweenUser_Part_{file_count}.txt")
    pool.apply_async(cosValue, args=(user_list_temp, Word2Vector_dic, candidate_shortID_set, out_file))
    pool.close()
    pool.join()
    print(f'实际用户数：{len(behavior_ID_list)}')
    print("Item相似度计算完成")


def loadCandidate(candidate_answer_file):
    candidate = []
    file_object = open(candidate_answer_file, 'r', encoding='UTF-8')
    for line in file_object:
        long_ID = line[:32]
        candidate.append(long_ID)
    print(f"候选集数量：{len(candidate)}")
    return candidate


def load_answer_dict(answer_id_dict_file):
    # 根据短ID 找长ID
    answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        tt = line.split('\t')
        answer_id_dict[tt[0]] = tt[1][:32]
    return answer_id_dict


def load_revers_answer_dict(answer_id_dict_file):
    # 根据长ID找短ID
    revers_answer_id_dict = dict()
    file_object = open(answer_id_dict_file, 'r', encoding='UTF-8')
    for line in file_object:
        revers_answer_id_dict[line.split('\t')[1][:32]] = line.split('\t')[0]
    return revers_answer_id_dict


def loadVector(Word2VectorFile, behavior_ID_set, candidate, answer_id_dict_file):
    revers_answer_id_dict = load_revers_answer_dict(answer_id_dict_file)
    candidate_shortID_set = set()
    for candi in candidate:
        if candi in revers_answer_id_dict.keys():
            candidate_shortID_set.add('A'+revers_answer_id_dict[candi])
    print(f"candidate_set数量为:{len(candidate_shortID_set)}")
    print(f"behavior_ID_set数量为:{len(candidate_shortID_set)}")
    behavior_ID_set = behavior_ID_set.union(candidate_shortID_set)
    print(f"behavior_ID_set合并更新后数量为:{len(behavior_ID_set)}")
    Word2Vector_dic = dict()
    temp = []
    file_object = open(Word2VectorFile, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split(" ")
        answer = t[0]
        if answer in behavior_ID_set:
            for num in t[1:len(t)]:
                temp.append(float(num))
            vector = np.array(temp)
            temp.clear()
            Word2Vector_dic[answer] = vector
    no_Vector = behavior_ID_set.difference(set(Word2Vector_dic.keys()))
    print(f"没有被映射成向量的问题ID数量:{len(no_Vector)}")
    print(no_Vector)
    print(f"词向量个数为:{len(Word2Vector_dic)}")
    c= 0
    for i in candidate_shortID_set:
        if i in Word2Vector_dic.keys():
            c +=1
    print(">>>>>>>>>>>>>>>>>>>>>" + str(c))
    return Word2Vector_dic, candidate_shortID_set


def cosValue(behavior_ID_list, Word2Vector_dic, candidate_shortID_set, outfile):
    name = multiprocessing.current_process().name
    print(f'线程{name}计算开始')
    Item_similary_dic = dict()
    file_object = open(outfile, 'w', encoding='UTF-8')
    for aid in behavior_ID_list:
        aid_list = dict()
        for ca in candidate_shortID_set:
            if aid in Word2Vector_dic.keys() and ca in Word2Vector_dic.keys():
                num = np.dot(Word2Vector_dic[aid], Word2Vector_dic[ca])
                denom = li.norm(Word2Vector_dic[aid]) * li.norm(Word2Vector_dic[ca])
                cos = num / denom
                aid_list[ca] = cos
        if len(aid_list) != 0:
            Item_similary_dic[aid] = sorted(aid_list.items(), key=lambda e: e[1], reverse=True)[:500]
    for key in Item_similary_dic.keys():
        file_object.write(key+"\t")
        for item in Item_similary_dic[key]:
            file_object.write(item[0]+":"+str(item[1])+",")
        file_object.write("\n")
    file_object.close()
    print(f'线程{name}计算完成')
    print(f"文件{outfile}打印完成")


if __name__ == '__main__':
    everyday_test_file = "E:\\CCIR_online\\20180809\\testing_set_20180808_RecSys_flappyBird.txt"
    candidate_answer_file = "E:\\CCIR_online\\20180809\\candidate_online_all.txt"
    answer_id_dict_file = "E:\\CCIR_online\\20180809\\answer_id_all.dict"
    Word2VectorFile = "E:\\CCIR_online\\20180809\\中间文件\\CCIR_online__ItemCF_0808.vec"
    out_Floder = "E:\\CCIR_online\\20180809\\中间文件\\Item_Similary"
    behavior_ID_set = load_every_test_read_answer(everyday_test_file)
    candidate = loadCandidate(candidate_answer_file)
    Word2Vector_dic, candidate_shortID_set = loadVector(Word2VectorFile, behavior_ID_set, candidate, answer_id_dict_file)
    multiprocessing_computer(3, behavior_ID_set, Word2Vector_dic, candidate_shortID_set, out_Floder)