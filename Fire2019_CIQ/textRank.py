# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 13:33
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : textRank.py
# @Software: PyCharm

from textrank4zh import TextRank4Keyword, TextRank4Sentence
import os
import jieba
import logging

# 取消jieba 的日志输出
# jieba.setLogLevel(logging.INFO)


def get_key_words(text, num=5):
    """提取关键词"""
    tr4w = TextRank4Keyword()
    tr4w.analyze(text, lower=True)
    key_words = tr4w.get_keywords(num)
    return [item.word for item in key_words]


# 命名 小行星 周先生 天文台 国际


def get_summary(text, num=3):
    """提取摘要"""
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    return [item.sentence for item in tr4s.get_key_sentences(num)]


def get_Object_casedocs_stemmer_topic_word(Object_casedocs_stemmer, Object_casedocs_stemmer_textRankTopic, topicWord=100):
    filelist = os.listdir(Object_casedocs_stemmer)
    for file in filelist:
        file_path = os.path.join(Object_casedocs_stemmer+"\\"+file)
        file_object = open(file_path, 'r', encoding='UTF-8')
        content = ""
        for line in file_object:
            content += line
        topicWords = get_key_words(content, topicWord)
        summarys = get_summary(text=content, num=topicWord)
        # print(topicWords)
        # print(summary)
        file_object.close()
        file_object_out = open(Object_casedocs_stemmer_textRankTopic+"\\"+file, 'w', encoding='UTF-8')
        for word in topicWords:
            file_object_out.write(word+" ")
        file_object_out.write("\t")
        for summary in summarys:
            file_object_out.write(summary+",")
        file_object_out.write("\n")
        file_object_out.close()
        print(file+"主题词提取完成")


def get_Query_doc_word_topic(Query_doc_stemmer, Query_doc_stemmer_textRankTopic, topcWord = 100):
    file_object = open(Query_doc_stemmer, 'r', encoding='UTF-8')
    file_object_out = open(Query_doc_stemmer_textRankTopic, 'w', encoding='UTF-8')
    for line in file_object:
        words = get_key_words(line, topcWord)
        summarys = get_summary(line, topcWord)
        for w in words:
            file_object_out.write(w+" ")
        file_object_out.write("\t")
        for summary in summarys:
            file_object_out.write(summary + " ")
        file_object_out.write("\n")
    file_object.close()
    file_object_out.close()



root_path = "E:\\Fire2019\\评测任务一\\AILA-data\\"
Object_casedocs_stemmer_floder = root_path + "Object_casedocs_stemmer"
Object_casedocs_stemmer_textRankTopic = root_path + "Object_casedocs_stemmer_textRankTopic"
Query_doc_stemmer = root_path + "\\Query_doc_stemmer.txt"
Query_doc_stemmer_textRankTopic = root_path + "\\Query_doc_stemmer_textRankTopic.txt"
# get_Object_casedocs_stemmer_topic_word(Object_casedocs_stemmer_floder, Object_casedocs_stemmer_textRankTopic, 100)
get_Query_doc_word_topic(Query_doc_stemmer, Query_doc_stemmer_textRankTopic, topcWord=100)