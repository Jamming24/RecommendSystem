# encoding:utf-8

import csv
import numpy as np
import gc
import math


def loadData(filepath):
    # 用于跳过第一行
    flag = False
    user_behavior = dict()
    csv_file = csv.reader(open(filepath, 'r'))
    for line in csv_file:
        if flag:
            if line[0] in user_behavior.keys():
                # 如果用户ID已经存在 就把观看的新电影 加入到观看列表中
                watchList = user_behavior[line[0]]
                if watchList.append(line[1]):
                    user_behavior[line[0]] = watchList
            else:
                watchList = [line[1]]
                user_behavior[line[0]] = watchList
        else:
            flag = True
            continue
    print('用户总量:' + str(len(user_behavior.keys())))
    # 返回含有用户行为信息的字典
    return user_behavior


def UserSimilarity(User_behavior):
    keys = User_behavior.keys()
    # 物品到用户的倒排表
    inverseTable = {}
    for kk in keys:
        Itemlist = User_behavior[kk]
        for item in Itemlist:
            if item in inverseTable.keys():
                userList = inverseTable[item]
                userList.append(kk)
                inverseTable[item] = userList
            else:
                userList = [kk]
                inverseTable[item] = userList
    # 物品到用户的倒排表生成完成
    # print(inverseTable)
    ItemLen = len(keys)
    # 回收内存
    gc.collect()
    # 创建和用户数相同维数的倒排矩阵 用来存放用户与用户之间购买相同的物品数
    inverseMatrix = np.zeros((ItemLen, ItemLen))
    for key in inverseTable.keys():
        value = inverseTable[key]
        for i in range(0, len(value) - 1):
            for j in range(i, len(value) - 1):
                row = int(value[i]) - 1
                col = int(value[j + 1]) - 1
                inverseMatrix[row][col] += 1
                inverseMatrix[col][row] += 1
    # print(inverseMatrix)
    # 计算用户与用户之间的相似度
    SimilarityWeight = np.zeros((ItemLen, ItemLen))
    for u in keys:
        for v in keys:
            row = int(u) - 1
            col = int(v) - 1
            if v != u and inverseMatrix[row][col] != 0.0:
                # 用户u和用户v 相同商品的个数 作为分子，（用户u和用户v的商品数的乘积）在开根号作为分母
                SimilarityWeight[row, col] = inverseMatrix[row][col] / math.sqrt(
                    len(User_behavior[u]) * len(User_behavior[v]) * 1.0)
    # print(SimilarityWeight)
    np.savetxt('E:\\迅雷下载\\ml-latest-small\\UserCF_Weight.txt', SimilarityWeight, fmt='%f', delimiter=' ', newline='\r\n')
    return SimilarityWeight


def Recommend(user, kuser, WeightMatrix, userbehavior):
    # 需要推荐的用户ID , 与用户兴趣最相关的k个用户, 用户之间的相似度衡量表, 用户行为数据
    ranklist = dict()
    relevantUser = dict()
    for Uid in range(0, len(WeightMatrix[int(user) - 1])):
        relevantUser[Uid + 1] = WeightMatrix[int(user) - 1][Uid]
    # 对字典进行按value排序
    relevantUser = sorted(relevantUser.items(), key=lambda e: e[1], reverse=True)
    # 重拍以后relevantUser字典变成list
    if kuser <= len(relevantUser):
        for index in range(0, kuser):
            # 列表中的元素relevantUser[index]为元组类型
            # 相关用户的ID
            relevantID = relevantUser[index][0]
            # 获取用户v的商品条目
            itemlist = userbehavior[str(relevantID)]
            for j in itemlist:
                if j in ranklist.keys():
                    ranklist[j] += WeightMatrix[int(user) - 1][relevantID - 1] * 1.0
                else:
                    ranklist[j] = WeightMatrix[int(user) - 1][relevantID - 1] * 1.0
    else:
        print('相关用户数量越界')

    # 去掉用户user买过的商品
    for item in userbehavior[str(user)]:
        if item in ranklist.keys():
            ranklist.pop(item)
        else:
            continue
    # 返回未排序的推荐结果字典，注意是未排序
    return ranklist


def PrintRecommendlist():
    filePath = 'E:\\迅雷下载\\ml-latest-small\\ratings.csv'
    User_behavior = loadData(filePath)
    # 用户ID 1-671
    recommendlist = Recommend('671', 15, UserSimilarity(User_behavior), User_behavior)
    # 将获得的推荐字典按照Value排序
    dic = sorted(recommendlist.items(), key=lambda e: e[1], reverse=True)
    # 打印前100个
    count = 0
    for k in dic:
        if count < 100:
            print(count+1, '>>>>', k)
            count += 1
