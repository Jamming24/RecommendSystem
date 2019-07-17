# encoding:utf-8
# 根据词频生成哈夫曼树

import snowballstemmer
import string

MAX_VALUE = 1000000


class treeNode:
    def __init__(self):
        self.name = '*'
        self.weight = 0
        self.left = -1
        self.right = -1
        self.parent = -1


def StaticalWordFrequently(train_data):
    WordFrequently = dict()
    file_object = open(train_data, 'r', encoding='UTF-8')
    content = ""
    for line in file_object:
        content += line
    without_punctuation = content.maketrans('', '', string.punctuation)
    str = content.translate(without_punctuation)
    Words = str.lower().split()
    print(f"总个数:{len(Words)}")
    Stemmer = snowballstemmer.EnglishStemmer()
    '''
    取词干之前：
        总个数:17005207
        总词数202003
    取词干之后：
        总个数:17005207
        总词数253854
    '''
    for w in Words:
        word = Stemmer.stemWord(word=w)
        # word =w
        if word in WordFrequently.keys():
            count = WordFrequently[word] + 1
            WordFrequently[word] = count
        else:
            WordFrequently[word] = 1
    print(f"总词数{len(WordFrequently)}")
    file_object.close()
    return WordFrequently


def PrintWordFrequently(WordFrequently, WordFile):
    file_objetct = open(WordFile, 'w', encoding='UTF-8')
    for word in WordFrequently.keys():
        file_objetct.write(word + "\t" + str(WordFrequently[word]) + "\n")
    file_objetct.close()


def createHuffmanTree(WordFrequentlyDict):
    HuffmanTree = []
    print("正在构建哈夫曼树")
    for word in WordFrequentlyDict.keys():
        newNode = treeNode()
        newNode.name = word
        newNode.weight = int(WordFrequentlyDict[word])
        newNode.left = -1
        newNode.right = -1
        newNode.parent = -1
        HuffmanTree.append(newNode)
    # 执行n-1次便可生成一个哈夫曼树
    n = len(WordFrequentlyDict)
    for i in range(n - 1):
        minFirst = minSecond = MAX_VALUE
        indexFirst = indexSecond = -1
        for j in range(n + i):
            if HuffmanTree[j].weight < minFirst and HuffmanTree[j].parent == -1:
                minSecond = minFirst
                indexSecond = indexFirst
                minFirst = HuffmanTree[j].weight
                indexFirst = j
            elif HuffmanTree[j].weight < minSecond and HuffmanTree[j].parent == -1:
                minSecond = HuffmanTree[j].weight
                indexSecond = j

        # 合并哈夫曼结点
        newParentNode = treeNode()
        newParentNode.weight = minFirst + minSecond
        newParentNode.left = indexFirst
        newParentNode.right = indexSecond
        if HuffmanTree[indexFirst].right == HuffmanTree[indexFirst].left == -1:
            newParentNode.left = HuffmanTree[indexFirst].name
        if HuffmanTree[indexSecond].right == HuffmanTree[indexSecond].left == -1:
            newParentNode.right = HuffmanTree[indexSecond].name
        HuffmanTree.append(newParentNode)
        newParentNode.name = len(HuffmanTree) - 1
        HuffmanTree[indexFirst].parent = n + i
        HuffmanTree[indexSecond].parent = n + i
    # 返回哈夫曼树
    print("哈夫曼树构建完成")
    return HuffmanTree


def loadWordFrequently(WordFrequentlyFile):
    WordFrequentlyDict = dict()
    file_object = open(WordFrequentlyFile, 'r', encoding='UTF-8')
    for line in file_object:
        WordAndFrequent = line.strip('\n').split('\t')
        WordFrequentlyDict[WordAndFrequent[0]] = WordAndFrequent[1]
    print(f"词表大小{len(WordFrequentlyDict)}")
    file_object.close()
    return WordFrequentlyDict


def getHuffmanCode(HuffmanTree, n, Outfile):
    # 返回哈夫曼编码
    print("正在生成哈夫曼编码")
    HaffmanCode = dict()
    # 编码栈
    CodeStack = []
    i = -1
    for currentNode in HuffmanTree:
        i += 1
        if i > n - 1:
            break
        if currentNode.left == currentNode.right == -1:
            local = currentNode.name
            parentIndex = currentNode.parent
            parentNodeleft = HuffmanTree[parentIndex].left
            parentNoderight = HuffmanTree[parentIndex].right
            while parentIndex != -1:
                if parentNodeleft == local:
                    CodeStack.append('0')
                    local = parentIndex
                    parentIndex = HuffmanTree[parentIndex].parent
                    parentNodeleft = HuffmanTree[parentIndex].left
                    parentNoderight = HuffmanTree[parentIndex].right
                elif parentNoderight == local:
                    CodeStack.append('1')
                    local = parentIndex
                    parentIndex = HuffmanTree[parentIndex].parent
                    parentNodeleft = HuffmanTree[parentIndex].left
                    parentNoderight = HuffmanTree[parentIndex].right
                else:
                    print("程序出错")
                    break
            code = "".join(CodeStack[::-1])
            CodeStack.clear()
            HaffmanCode[currentNode.name] = code
    file_object = open(Outfile, 'w', encoding='UTF-8')
    for word in HaffmanCode.keys():
        file_object.write(word + "\t" + str(HaffmanCode[word]) + "\n")
    file_object.close()
    print("哈夫曼编码计算完成")


def PrintHuffmanTree(HuffmanTree):
    print("k\t\tWeight\tParent\tLchild\tRchild")
    index = 0
    for i in HuffmanTree:
        print(f"{i.name}\t\t{i.weight}\t\t{i.parent}\t\t{i.left}\t\t{i.right}")
        index += 1


# 统计词频
# WordFrequently = StaticalWordFrequently(train_data="./English_data.txt")
# PrintWordFrequently(WordFrequently, "./WordFrequently.txt")
# 生成哈夫曼树和哈夫曼编码
# WordFrequentlyDict = loadWordFrequently("./WordFrequently.txt")
# HuffmanTree = createHuffmanTree(WordFrequentlyDict)
# PrintHuffmanTree(HuffmanTree)
# getHuffmanCode(HuffmanTree, len(WordFrequentlyDict), "./HuffmanNode.txt")


'''
k               Weight          Parent          Lchild          Rchild
0               23              6               -1              -1
1               45              7               -1              -1
2               78              8               -1              -1
3               45              8               -1              -1
4               90              9               -1              -1
5               12              6               -1              -1
6               35              7               5               0
7               80              9               6               1
8               123             10              3               2
9               170             10              7               4
10              293             -1              8               9
'''
