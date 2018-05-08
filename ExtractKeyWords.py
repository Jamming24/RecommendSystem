# -*- coding:utf-8 -*-

import jieba
import re


def getKeyWords(NewTopicList):
    keysWordlist = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for news in NewTopicList:
        line = re.sub(r, '', news.split(',')[1])
        seg_list = jieba.cut(line, cut_all=False)
        keys = '/ '.join(seg_list)
        # print('精确模式Default:', keys)  # 精确模式
        keysWordlist.append(keys)
        # seg_list = jieba.cut(news.split(',')[1], cut_all=True)
        # print('全模式:', '/ '.join(seg_list))  # 全模式
        # seg_list = jieba.cut_for_search(news.split(',')[1])  # 搜索引擎模式
    return keysWordlist


def loadHotPotNews(filepath):
    newslist = []
    # r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    f = open(filepath, 'r')
    for line in f:
        # line = re.sub(r, '', line)
        print(line)
        newslist.append(line.strip())
    f.close()
    return newslist


# filePath = 'C:\\Users\\Jamming\\Desktop\\表\\Hotpot_News.txt'

# getKeyWords(loadHotPotNews(filePath))
