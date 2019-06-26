# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 18:34
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : get_wordPOS_feature.py
# @Software: PyCharm
import codecs
import json
import jieba.analyse
import jieba.posseg as pseg
from tqdm import tqdm


def get_topK_wordPOS(content, topk):
    # 返回输入进来的文章或者标题的topk个词的词性特征 词性包括名词，动词，形容词，副词
    # n:1,0,0,0  v:0,1,0,0  a:0,0,1,0  d:0,0,0,1
    topK_words = jieba.analyse.extract_tags(sentence=content, topK=topk)
    words = pseg.cut(content)
    words_POS_tag = {}
    for w in words:
        if w.word in topK_words:
            if w.flag == 'n':
                words_POS_tag[w.word] = [1, 0, 0, 0]
            elif w.flag == 'v':
                words_POS_tag[w.word] = [0, 1, 0, 0]
            elif w.flag == 'a':
                words_POS_tag[w.word] = [0, 0, 1, 0]
            elif w.flag == 'd':
                words_POS_tag[w.word] = [0, 0, 0, 1]
    return words_POS_tag


def load_json_data(file_path):
    print('load file', file_path)
    f = codecs.open(file_path, 'r', 'utf-8')
    data = {}
    for line in tqdm(f.readlines()):
        news = json.loads(line.strip())
        data[news['newsId']] = news
    return data


def write_txt_data(file_path, data):
    print('write file', file_path)
    with open(file_path, 'w', encoding='utf-8') as fw:
        for item in tqdm(data):
            fw.write(item + '\n')


def print_POS_tag_feature(data_source, out_path, topK):
    items = []
    for newsId in tqdm(data_source.keys()):
        source_item = data_source[newsId]
        # title = source_item['title']
        content = source_item['content'].replace('\n', '')
        words_POS_tag = get_topK_wordPOS(content, topK)
        feature = ""
        for word in words_POS_tag.keys():
            feature += word + ":"+str(words_POS_tag[word])
        items.append('%s\t%s' % (newsId, feature))
    write_txt_data(out_path, items)


file_path = "C:/Users/Jamming/Desktop/souhu/sohu_data/data/data_v2/coreEntityEmotion_train.txt"
out_path = "C:/Users/Jamming/Desktop/souhu/POS_tag_feature.txt"
topK = 30
data_source = load_json_data(file_path)
print_POS_tag_feature(data_source, topK=topK, out_path=out_path)

