# encoding:utf-8
# 计算用户与用户之间的相似度，输出到与之前相似的
# 本程序功能 计算用户之间的相似度 排序取前100个 按照过去的格式 输出到文件夹即可
import numpy as np
import numpy.linalg as li
import multiprocessing


def load_small_test(small_test_file):
    test_ID = []
    file_object = open(small_test_file, 'r', encoding='UTF-8')
    for line in file_object:
        userID = line.split("\t")[0]
        test_ID.append(userID)
    file_object.close()
    print(f"用户ID数量{len(test_ID)}")
    return test_ID


def load_UserVector(UserVector_File):
    User_Word2Vector_dic = dict()
    temp = []
    file_object = open(UserVector_File, 'r', encoding='UTF-8')
    for line in file_object:
        t = line.split(" ")
        answer = t[0]
        for num in t[1:len(t)]:
            temp.append(float(num))
        vector = np.array(temp)
        temp.clear()
        User_Word2Vector_dic[answer] = vector
    print(f"用户向量个数为:{len(User_Word2Vector_dic)}")
    file_object.close()
    return User_Word2Vector_dic


def Computataional_User_Similaryity(part_test_user_ID, User_Word2Vector_dic):
    # 计算13.5万名用户与534万用户之间的相似度，在计算过程中 直接需要计算 13.5万x521万次
    # 计算相似度的过程中 如果分子为零 则直接跳过计算，节省时间
    name = multiprocessing.current_process().name
    print(f"进程{name}开始执行")
    part_User_Similary_dict = dict()
    for userID in part_test_user_ID:
        if userID in User_Word2Vector_dic.keys():
            simply_similary_dic = dict()
            Vector_one = User_Word2Vector_dic[userID]
            for id in User_Word2Vector_dic.keys():
                if userID != id:
                    Vector_two = User_Word2Vector_dic[id]
                    num = np.dot(Vector_one, Vector_two)
                    if num != 0:
                        denom = li.norm(Vector_one) * li.norm(Vector_two)
                        cos = num / denom
                        simply_similary_dic[id] = cos
            simply_similary_list = sorted(simply_similary_dic.items(), key=lambda e: e[1], reverse=True)[:500]
            if len(simply_similary_list) != 0:
                part_User_Similary_dict[userID] = simply_similary_list
    print(f"进程{name}运算完成")
    del part_test_user_ID
    del User_Word2Vector_dic
    return part_User_Similary_dict


def multiprocessing_manager(test_user_ID, User_Word2Vector_dic, n):
    User_Similary_dict = dict()
    pool = multiprocessing.Pool(processes=n)
    result_list = []
    temp = []
    for UserID in test_user_ID:
        if len(temp) == 500:
            part_test_user_ID = temp.copy()
            temp_User_Word2Vector_dic = User_Word2Vector_dic.copy()
            result = pool.apply_async(Computataional_User_Similaryity, args=(part_test_user_ID, temp_User_Word2Vector_dic))
            result_list.append(result)
            temp.clear()
        else:
            temp.append(UserID)
    print(f"最后一部分用户数量{len(temp)}")
    result = pool.apply_async(Computataional_User_Similaryity, args=(temp, User_Word2Vector_dic))
    result_list.append(result)
    pool.close()
    pool.join()
    del test_user_ID
    del User_Word2Vector_dic
    for res in result_list:
        User_Similary_dict.update(res.get())
    print(f"计算用户数量:{len(User_Similary_dict)}")
    print("用户相似度计算完成,Yeah!!!!!!")

    return User_Similary_dict


def Print_User_Similary(User_Similary_dict, User_Similary_Outfile):
    Users_Similary_file_object = open(User_Similary_Outfile, 'w', encoding='UTF-8')
    for userID in User_Similary_dict:
        Users_Similary_file_object.write(userID + " ")
        for item in User_Similary_dict[userID]:
            line = str(item[0]) + "," + str(item[1]) + " "
            Users_Similary_file_object.write(line)
        Users_Similary_file_object.write("\n")
    Users_Similary_file_object.close()
    print("相似度文件输出完成")


if __name__ == '__main__':
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    UserVector_File = "E:\\CCIR\\small_CCIR_train_test_UserCF.vec"
    User_Similary_Outfile = "E:\\CCIR\\UserCF_UserSimilary_baseOnWord2Vector.txt"
    # n表示进程池最大进程数量
    n = 5
    test_ID = load_small_test(small_test_file)
    User_Word2Vector_dic = load_UserVector(UserVector_File)
    # part_User_Similary_dict = Computataional_User_Similaryity(test_ID, User_Word2Vector_dic)
    User_Similary_dict = multiprocessing_manager(test_ID, User_Word2Vector_dic, n)
    Print_User_Similary(User_Similary_dict, User_Similary_Outfile)