# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 14:59
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : gensim_Word2Vector.py
# @Software: PyCharm

import logging
from gensim.models import word2vec


def train_word2vector(wiki_sentence, vector_file):

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    model_100 = word2vec.Word2Vec(word2vec.LineSentence(wiki_sentence), size=200, window=11, min_count=5, negative=10,
                                  sg=0, hs=0, workers=2, alpha=0.025)
    model_100.wv.save_word2vec_format(vector_file, binary=False)


zh_wiki = "E:/Teacher_Qi/zh_wiki_sentence_simplified.txt"
vector = "E:/Teacher_Qi/zh_wiki_sentence_simplified_Vector.txt"
train_word2vector(zh_wiki, vector)
