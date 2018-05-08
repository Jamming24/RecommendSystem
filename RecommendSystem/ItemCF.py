# encoding:utf-8

import csv
import numpy as np
import UserCF
import sys
import math


def ItemSimilarity(userbehavior_data, IteMapId):
    # 建立字典存储每个商品对应的购买用户数
    ItemUserNum = dict()
    # 用户列表user_keys
    user_keys = userbehavior_data.keys()
    # 物品相似度矩阵 Item_Matrix 暂时存为整数  未做归一化处理
    Item_Matrix = np.zeros((len(IteMapId.keys()), len(IteMapId.keys())))
    for user in user_keys:
        print('正在计算用户', user)
        behaviorlist = userbehavior_data[user]
        for index in range(0, len(behaviorlist)):
            # 遍历过程中  统计每个Item被使用或者购买的情况
            if behaviorlist[index] in ItemUserNum:
                ItemUserNum[behaviorlist[index]] += 1
            else:
                ItemUserNum[behaviorlist[index]] = 1
            #######################################
            # 计算得到物品相似度矩阵
            for j in range(index + 1, len(behaviorlist)):
                row = IteMapId[behaviorlist[index]]
                col = IteMapId[behaviorlist[j]]
                Item_Matrix[row][col] += 1
                Item_Matrix[col][row] += 1
            ###################
    # print(ItemUserNum)
    print('Item_Matrix(物品被同时购买次数矩阵)占用空间:', sys.getsizeof(Item_Matrix))
    np.savetxt('E:\\迅雷下载\\ml-latest-small\\ItemCF_Weight.txt', Item_Matrix, fmt='%f', delimiter=' ',
               newline='\r\n')
    return ItemUserNum, Item_Matrix


def Recommend(user, kuser, ItemUserNum, Item_Matrix, userbehavior_Data, ItemMapID, UserBehaviorWeight):
    Ranklist = dict()
    templist = dict()
    # 为了优化时间和空间复杂度, 物品与物品间的相似度将会在此模块中进行计算
    # 推荐用户Id, k个最相似物品, 每个物品被购买次数字典, 物品被同时购买次数矩阵, 用户行为数据, 物品和ID映射的MAP, 用户对电影评价字典
    # 获取用户喜爱商品列表
    userFavoriteList = userbehavior_Data[user]
    for item in userFavoriteList:
        # 获取与这个商品相似的物品列表
        # print(f'计算物品{item}与其他物品的相似度')
        similarylist = Item_Matrix[ItemMapID[item]]
        index = 0
        # 将相似物品列表转存为字典 便于排序，取前n个
        for relevantItem in ItemMapID.keys():
            # 分子numerator表示的是物品u和v被不同用户共同购买的次数
            numerator = similarylist[index]
            x = ItemUserNum[item]
            # 分母denominator表示的是根号下（物品u被购买的次数*物品v被购买的次数）
            # 如果有电影没有被观看过 这里会发生索引错误 需要特殊处理
            try:
                y = ItemUserNum[relevantItem]
            except KeyError:
                print(f'编号为{relevantItem}的电影没有被观看记录')
                templist[relevantItem] = 0.0
            else:
                denominator = math.sqrt(x * y)
                templist[relevantItem] = numerator / denominator
            index += 1
        # 对templist字典进行重排得到元组
        Sortlist = sorted(templist.items(), key=lambda e: e[1], reverse=True)
        templist.clear()
        ######################################
        # 在引用用户对自己喜欢的物品进行打分的数据之后 可以在此处进行优化，将用户对买过物品的打分与物品之间相似度乘积加和即可
        # 如果不引入用户对购买过的商品的打分数据 那么基于ItemCF算法性能无法完全发挥，那么下面的代码也是简单的重拍序而已
        # 这样为用户推荐的商品偏向于热门商品，容易造成马太效应
        # 在进行物品之间相似度计算时 可以引入最大值归一化原来 可以提高推荐准确度和覆盖率
        # print('商品', item, '的相关物品以及权重', Sortlist)
        for kk in range(0, kuser):
            # 取前k个与用户喜爱物品最相关的物品进行计算
            if Sortlist[kk][0] in userFavoriteList:
                # 用户购买过的商品不在进行计算，用来节省时间和内存
                continue
            elif Sortlist[kk][0] not in Ranklist.keys():
                Ranklist[Sortlist[kk][0]] = float(UserBehaviorWeight[user][item]) * Sortlist[kk][1]
            else:
                Ranklist[Sortlist[kk][0]] += (float(UserBehaviorWeight[user][item]) * Sortlist[kk][1])
        ######################################

    return user, Ranklist


def PrintRecommendList(user, RankList, Item_Dic, n):
    # 用户未排序的推荐列表 计算ID与电影ID的映射Map  打印前n个
    print(f'用户{user}的推荐列表中前{n}名')
    Sort_ranktuple = sorted(RankList.items(), key=lambda e: e[1], reverse=True)
    # 打印 计算ID，电影详情(表中ID,电影名称,类型)，计算得分
    for i in range(n):
        print(Sort_ranktuple[i][0], '>>>>', Item_Dic[int(Sort_ranktuple[i][0])], '>>>>', Sort_ranktuple[i][1])
    print('推荐完成')


def loadItemData(csvfilePath):
    # 将电影信息存储到字典里
    Item_Dic = dict()
    # 建立矩阵索引和电影Id的映射表，便于存取
    ItemMapingID = dict()
    csv_file = csv.reader(open(csvfilePath, 'r'))
    flag = False
    index = 0
    for movie in csv_file:
        if flag:
            Item_Dic[index] = movie
            ItemMapingID[movie[0]] = index
            index += 1
        else:
            flag = True
            continue
    return Item_Dic, ItemMapingID


def loadUserWeight(csvfile):
    UserBehaviorWeight = dict()
    csvfileData = csv.reader(open(csvfile, 'r'))
    flag = False
    for user in csvfileData:
        if flag:
            # UserBehaviorWeight[user[0]] = user[]
            if user[0] not in UserBehaviorWeight.keys():
                ratingdic = dict()
                ratingdic[user[1]] = user[2]
                UserBehaviorWeight[user[0]] = ratingdic

            else:
                ratingdic = UserBehaviorWeight[user[0]]
                ratingdic[user[1]] = user[2]
        else:
            flag = True
            continue

    # for i in UserBehaviorWeight.keys():
    #     print(f'用户{i}的电影评分')
    #     for j in UserBehaviorWeight[i].keys():
    #         print(f'电影{j},评分{UserBehaviorWeight[i][j]}')

    return UserBehaviorWeight


# 测试小例子
# User = {'1': ['a', 'b', 'd'], '2': ['b', 'c', 'e'], '3': ['c', 'd'], '4': ['b', 'c', 'd'], '5': ['a', 'd']}
# Userlist = ['a', 'b', 'c', 'd', 'e']
# Userbehavoirating = {'1': {'a': '5', 'b': '4', 'd': '2'}, '2': {'b': '2', 'c': '4', 'e': '1'}, '3': {'c': '3', 'd': '4'},
#                 '4': {'b': '1', 'c': '3', 'd': '4'}, '5': {'a': '1', 'd': '5'}}
# testIteMapId = dict()
# for i in range(0, 5):
#     testIteMapId[Userlist[i]] = i
# Matrix = ItemSimilarity(User, testIteMapId)
# Recommend('1', 3, Matrix[0], Matrix[1], User, testIteMapId, Userbehavoirating)

filePath = 'E:\\迅雷下载\\ml-latest-small\\ratings.csv'
movie_path = 'E:\\迅雷下载\\ml-latest-small\\movies.csv'
data = loadItemData(movie_path)
IdMapingName = data[0]
userbehavior_rating = loadUserWeight(filePath)
userbehavior = UserCF.loadData(filePath)
IteMatrix = ItemSimilarity(userbehavior, data[1])
SortList = Recommend('190', 10, IteMatrix[0], IteMatrix[1], userbehavior, data[1], userbehavior_rating)
PrintRecommendList(SortList[0], SortList[1], IdMapingName, 20)
