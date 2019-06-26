# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 9:22
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : get_sentence_emotion_train_data.py
# @Software: PyCharm

import codecs
import json
from tqdm import tqdm

# 训练句子情感模型 取语料中包含实体词的前五句作为训练语料  max_seq_length=64


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


def get_sentence_emotion(data_source, topSentence, out_path):
    emotion_sentence = ['newsId\ttitle\tsentence\tlabel\tentity_word']
    for newsId in tqdm(data_source.keys()):
        source_item = data_source[newsId]
        title = source_item['title'].replace('\n', '').replace('\t', '')
        content = source_item['content'].replace('\n', '')
        contents = content.split("。")
        Emotions = source_item["coreEntityEmotions"]
        for item in Emotions:
            emotion_sentence_nums = 0
            entity_word = item['entity']
            label = item['emotion']
            for sentence in contents:
                if entity_word in sentence and emotion_sentence_nums < topSentence:
                    emotion_sentence.append('%s\t%s\t%s\t%s\t%s' % (newsId, title, sentence, label, entity_word))
    write_txt_data(out_path, emotion_sentence)


def get_sentence_title_and_sentence(data_source, out_path):
    emotion_sentence = ['newsId\ttitle\tsentence']
    for newsId in tqdm(data_source.keys()):
        topSentence = 20
        source_item = data_source[newsId]
        title = source_item['title'].replace('\n', '').replace('\t', '')
        content = source_item['content'].replace('\n', '').replace('\t', '')
        contents = content.split("。")
        if len(contents) < topSentence:
            topSentence = len(contents)
        #######存在bug 导致单句的文章丢失#########
        for index in range(topSentence):
        ######################################
            if contents[index] == "":
                continue
            else:
                emotion_sentence.append('%s\t%s\t%s' % (newsId, title, contents[index]))
    write_txt_data(out_path, emotion_sentence)


root_path = "C:/Users/Jamming/Desktop/souhu"
file_path = root_path+"/"+"sohu_data/data/data_v2/coreEntityEmotion_train.txt"
train_data_path = root_path+"/" + "HGongC/candidate_corener_emotion/data/emotion/all_sentence_emotion_train_data.tsv"
topSentence = 20
# data_source = load_json_data(file_path)
sentence_title_out_path = root_path+"/" + "HGongC/candidate_corener_emotion/data/emotion/title_sentence_data.tsv"
# get_sentence_emotion(data_source, topSentence, out_path=train_data_path)
# get_sentence_title_and_sentence(data_source, out_path=sentence_title_out_path)

stage2_test = root_path+"/coreEntityEmotion_test_stage2/coreEntityEmotion_test_stage2.txt"
stage2_test_title_sentence = root_path+"/coreEntityEmotion_test_stage2/8W_title_sentence_test_stage2.txt"
data_source = load_json_data(stage2_test)
get_sentence_title_and_sentence(data_source, stage2_test_title_sentence)

