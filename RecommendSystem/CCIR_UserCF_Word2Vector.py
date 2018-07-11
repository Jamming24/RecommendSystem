# encoding:utf-8
# 计算用户与用户之间的相似度，输出到与之前相似的
# 本程序功能 计算用户之间的相似度 排序取前100个 按照过去的格式 输出到文件夹即可


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
    UserVector_dic = dict()
    file_object = open(UserVector_File, 'r', encoding='UTF-8')
    all_line = file_object.readlines()
    for line in all_line:
        similary_list = []
        t = line.split('\t')
        Item_ID = t[0]
        for index in range(1, len(t) - 1):
            similary_list.append(t[index].split('(')[1].split(',')[0])
        UserVector_dic[Item_ID] = similary_list
    file_object.close()


def Computataional_User_Similaryity(testID, UserVector_File):
    # 计算13.5万名用户与534万用户之间的相似度，在计算过程中 直接需要计算 13.5万x521万次
    # 计算相似度的过程中 如果分子为零 则直接跳过计算，节省时间
    for userID in test_ID:
        print(userID)


if __name__ == '__main__':
    small_test_file = "E:\\CCIR\\testing_set_135089.txt"
    UserVector_File = dict()
    test_ID = load_small_test(small_test_file)
    Computataional_User_Similaryity(test_ID, UserVector_File)