# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import Word2Vec
import logging
import multiprocessing
import re
import sys
from gensim.models.word2vec import LineSentence

# r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0-9]+'
r = '[^a-zA-Z ]'  # 仅保留英文字符和空格
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))#13:00开始学习


def load_marp():
    content = []
    lines = []

    with open('../data/msrp_all.txt', 'r', encoding='utf-8') as fr:  # 加载原始语料数据
        for line in fr:  # 这里取前十万条数据^^^^^^^^^^^^^^^^^^^^^^^^^^^________________________________^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # line = fr.readline()
            lines.append(line)
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            # print(message[0], " ", message[1])
            content.append(re.sub(r, '', message[3]))
            content.append(re.sub(r, '', message[4]))
    print('数据加载完毕。。。')
    return content


def load_wiki():
    wiki_content = []
    with open(r'D:\tianliuya\wiki_google\google_sen.txt', 'r', encoding='utf-8') as fr:
        for line in fr:
            wiki_content.append(line)
    return wiki_content


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    # content_junk = load_marp()
    # content_wiki = load_wiki()
    # content_all = content_junk + content_wiki
    # seglist_all = []
    # file_split = open('../data/temp_300.txt', 'w', encoding='utf-8')  # 生成中间结果，保存中文分词后的结果
    # for line in content_all:
    #     file_split.writelines(re.sub(r, '', line.lower()).strip() + '\n')
    #     seglist_all.append(line)
    # file_split.close()



    model_out = 'D:\\CCIR\\CCIR_train_set_w2c.model'  # 生成并保存向量模型地址
    vec_out = 'D:\\CCIR\\CCIR_train_set.vec'  # 保存单词向量地址

    model = Word2Vec(LineSentence("D:\\CCIR\\word2vector_training_set.txt"), size=300, window=8, min_count=5,
                     workers=multiprocessing.cpu_count())
    # model.save(model_out)
    model.wv.save_word2vec_format(vec_out, binary=False)
