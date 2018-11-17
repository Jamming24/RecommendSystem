# encoding:utf-8

import numpy as np
import random
import HuffmanTree
import snowballstemmer
import math
# 提交晚了

def loadCorpusData(dataFile, sliding_window):
    train_data = []
    file_object = open(dataFile, 'r', encoding='UTF-8')
    content = ""
    for line in file_object:
        content += line
    Words = content.split()
    for index in range(sliding_window, len(Words) - sliding_window):
        # print(Words[index-sliding_window:index+sliding_window+1])
        # print('输出字符：', Words[index])
        train_data.append(Words[index - sliding_window:index + sliding_window + 1])
    file_object.close()
    print(f"数据总量{len(train_data)}")
    return train_data


def getHuffmanCode_update(HuffmanTree_dict, codeName):
    # 查找指定元素的哈夫曼编码，以及对应结点的参数值,
    # 由于得到的哈夫曼编码是倒叙输出的，同时参数向量也是倒叙找到的
    # 所以在进行返回结果的时候要进行逆置
    # print(f"正在查找{codeName}哈夫曼编码")
    currentNode = HuffmanTree_dict[codeName]
    # 编码栈
    CodeStack = []
    parentNode = []
    if currentNode.left == currentNode.right == -1:
        local = currentNode.name
        parentIndex = currentNode.parent
        parentNode.append(parentIndex)
        parentNodeleft = HuffmanTree_dict[parentIndex].left
        parentNoderight = HuffmanTree_dict[parentIndex].right
        while parentIndex != -1:
            if parentNodeleft == local:
                CodeStack.append('0')
            elif parentNoderight == local:
                CodeStack.append('1')
            else:
                print("程序出错")
                break
            local = parentIndex
            parentIndex = HuffmanTree_dict[parentIndex].parent
            parentNode.append(parentIndex)
            if parentIndex in HuffmanTree_dict.keys():
                parentNodeleft = HuffmanTree_dict[parentIndex].left
                parentNoderight = HuffmanTree_dict[parentIndex].right
        code = "".join(CodeStack[::-1])
        CodeStack.clear()
        # print(parentNode[::-1])
        # 返回结点的haffman编码和经过父节点序号
        return code, parentNode[::-1]


def trainWord2Vector(nVector, HuffmanTree_list, trainData):
    count = 0
    # 学习速率 和平均误差率
    learn_rate = 0.05
    mean_error = 0.0
    Stemmer = snowballstemmer.EnglishStemmer()
    # 将huffman树改成基于字典结构的，便于快速查找所需要子树
    # 存储词向量的字典
    Word_X = dict()
    Parameter_W = dict()
    HuffmanTree_dict = dict()
    for index in range(len(HuffmanTree_list)):
        node = HuffmanTree_list[index]
        HuffmanTree_dict[node.name] = node
    # 实现随机梯度下降
    # 训练数据个数
    n = len(trainData)
    for index in range(n):
        count += 1
        if count % 5000 == 0:
            print(f"已经训练了{count}数据")
        # 获取当前训练集合中训练数量
        current_num = len(trainData)
        # 随机选择训练数据, 计算完成之后将训练用过的数据移除，并随机挑选新的训练数据，以达到随机梯度下降的效果
        data_index = random.randint(0, current_num - 1)
        # print(data_index, '当前数据：', trainData[data_index])
        Simply_data = trainData[data_index]
        sum_Vector = np.zeros(nVector)
        midde = int(len(Simply_data) / 2)
        # codename为训练数据中最中间的词
        codeName = Stemmer.stemWord(Simply_data[midde])
        del Simply_data[midde]
        # print(Simply_data)
        for ws in Simply_data:
            w = Stemmer.stemWord(ws)
            if w in Word_X.keys():
                sum_Vector += Word_X[w]
            else:
                # 随机为新词初始化一个n维数组
                X = (np.random.random(nVector)-0.5)*2
                sum_Vector += X
                Word_X[w] = X
        huffmanCode, ParentCode = getHuffmanCode_update(HuffmanTree_dict, codeName)
        # print(huffmanCode, ':::', ParentCode[1:])
        # 获取各个父节点所包含的参数向量，如果没有要随机生成
        current_Parameter_W_dict = dict()
        for i in range(1, len(ParentCode)):
            pc = ParentCode[i]
            if pc in Parameter_W.keys():
                current_Parameter_W_dict[pc] = Parameter_W[pc]
            else:
                # 向量初始化-1 到 1之间
                W = (np.random.random(nVector)-0.5)*2
                Parameter_W[pc] = W
                current_Parameter_W_dict[pc] = W
        #######################################
        e = np.zeros(nVector)
        code_index = 0
        for code in ParentCode[1:]:
            # q 表示向量X与结点参数parameter的乘积，并带入sigmoid函数转化为概率
            parameter_w = current_Parameter_W_dict[code]
            martix_dot = round(float(np.dot(sum_Vector, parameter_w)), 5)
            # q = round(1 / (1+(round(math.exp(-martix_dot), 6))), 10)
            # try:
            #
            #     q = 1 / (1 + (1 / math.exp(- martix_dot)))
            # except OverflowError:
            #     print('数值太大')
            #     break
            # except ZeroDivisionError:
            #     print('捕获处零异常')
            #     break
            if martix_dot >= 20:
                q = 1.0
            elif martix_dot <= -20:
                q = 0.0
            else:
                q = 1 / (1 + (1 / math.exp(- martix_dot)))
            # 根据哈夫曼编码判断是正类还是负类
            code_class = int(huffmanCode[code_index])
            # 表示偏导数的计算公式之前的系数
            g = learn_rate * (1 - code_class - q)
            # e表示学习速率*偏导数在求和之和的矩阵
            e += g * np.array(parameter_w)
            # 对参数执行梯度下降
            parameter_w += g * np.array(sum_Vector)
            code_index += 1
            current_Parameter_W_dict[code] = parameter_w
        for ws in Simply_data:
            w = Stemmer.stemWord(ws)
            x_vector = Word_X[w]
            x_vector += e
            Word_X[w] = x_vector
            # print(x_vector)
        del trainData[data_index]
    print("训练完成")
    return Word_X


def PrintWord_Vector(Word_X, Outfile):
    file_object = open(Outfile, 'w', encoding='UTF-8')
    for w in Word_X.keys():
        file_object.write(w+':')
        for num in Word_X[w]:
            file_object.write(str(round(num, 5))+',')
        file_object.write('\n')
    file_object.close()


dataFile = "E:\\text.txt"
trainData = loadCorpusData(dataFile, 4)
wfd = HuffmanTree.loadWordFrequently(WordFrequentlyFile="E:\\WordFrequently.txt")
HuffmanTree_list = HuffmanTree.createHuffmanTree(WordFrequentlyDict=wfd)
Word_X = trainWord2Vector(10, HuffmanTree_list, trainData)
PrintWord_Vector(Word_X, "E:\\Word_Vector.txt")
