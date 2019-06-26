# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 17:23
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : commit_emotion.py
# @Software: PyCharm

import codecs
import json
from tqdm import tqdm


def get_test_id(test_stage2_path):
    # 返回文章ID列表
    ID_list = []
    flag = True
    test_stage2_path = root_path+"/"+test_stage2_path
    file_object = open(test_stage2_path, 'r', encoding='UTF-8')
    for line in file_object:
        if flag:
            flag = False
            continue
        ID_list.append(line.split("\t")[0])
    file_object.close()
    return ID_list


def write_txt_data(file_path, data):
    print('write file', file_path)
    with open(file_path, 'w', encoding='utf-8') as fw:
        for item in tqdm(data):
            fw.write(item + '\n')


def write_json_data(file_path, data):
    print('write file', file_path)
    with open(file_path, 'w', encoding='utf-8') as fw:
        for item in tqdm(data):
            fw.write(json.dumps(item) + '\n')


def get_emotion_result(bert_emotion):
    score_list = []
    bert_emotion = root_path+"/"+bert_emotion
    file_object = open(bert_emotion, 'r', encoding='UTF-8')
    for line in file_object:
        scores = line.split("\t")
        scores = list(map(eval, scores))
        emotion = get_high_score(scores)
        score_list.append(emotion)
    file_object.close()
    return score_list


def get_emotion_result_value(bert_emotion):
    # 文件顺序是POS NORM NEG
    item_list = []
    bert_emotion = root_path+"/"+bert_emotion
    file_object = open(bert_emotion, 'r', encoding='UTF-8')
    for line in file_object:
        scores = line.strip("\n").split("\t")
        item_list.append('"POS":'+scores[0]+',"NEG":'+scores[2]+',"NORM":'+scores[1])
    file_object.close()
    return item_list


def get_high_score(score_list):
    # 文件顺序是POS NORM NEG
    t = score_list.index(max(score_list))
    if t == 0:
        return "POS"
    elif t == 1:
        return "NORM"
    else:
        return "NEG"


def print_commit_reslut(output_flie, ID_list, score_list):
    output_flie = root_path+"/"+output_flie
    file_object = open(output_flie, 'w', encoding='UTF-8')
    for index in range(len(ID_list)):
        file_object.write(ID_list[index]+"\t"+score_list[index]+"\n")
    file_object.close()
    print("写入成功")


def bert_result_to_json(output_flie_json, ID_list, item_list):
    # file_object = open(root_path+"/"+output_flie_json, 'w', encoding='UTF-8')
    file_object = open(output_flie_json, 'w', encoding='UTF-8')
    for index in range(len(ID_list)):
        file_object.write('{"newsId":"'+ID_list[index]+'",'+item_list[index]+'}\n')
    file_object.close()
    print("json文件写入成功")


def load_trainID(train_path):
    trainID = []
    print('load file', train_path)
    f = codecs.open(train_path, 'r', 'utf-8')
    for line in tqdm(f.readlines()):
        news = json.loads(line.strip())
        trainID.append(news['newsId'])
    return trainID


def title_sentence_bert_result_json(path_result, trainID, title_emotion_result, title_sentence_emotion_result, title_sentence_test_file):
    json_items = []
    title_sentence_file_object = open(title_sentence_test_file, 'r', encoding='UTF-8')
    title_emotion_file_object = open(title_emotion_result, 'r', encoding='UTF-8')
    title_sentence_emotion_file_object = open(title_sentence_emotion_result, 'r', encoding='UTF-8')
    title_score = {}
    index = 0
    # 文件顺序是POS NORM NEG
    for title_line in title_emotion_file_object:
        titles = title_line.replace("\n", "").split("\t")
        title_score[trainID[index]] = "{'POS':"+titles[0]+",'NEG':"+titles[2]+",'NORM':"+titles[1]+"}"
        index += 1
    print(f"title_score:{len(title_score)}")
    title_sentence_score = []
    for item in title_sentence_emotion_file_object:
        items = item.replace("\n", "").split("\t")
        title_sentence_score.append("{'POS':"+items[0]+",'NEG':"+items[2]+",'NORM':"+items[1]+"}")
    print(f"title_sentence_score:{len(title_sentence_score)}")
    item_index = 0
    flag = True
    Print_Item = {}
    sentence_score = {}
    newID_title = {}
    test = set()
    for line in title_sentence_file_object:
        if flag:
            flag = False
            continue
        contents = line.replace("\n", "").split("\t")
        newID = contents[0]
        title = contents[1]
        newID_title[newID] = title
        sentence = contents[2]
        test.add(newID)
        if newID in Print_Item.keys():
            temp = Print_Item[newID]
            temp.append(sentence)
            Print_Item[newID] = temp
            sentence_score[sentence] = title_sentence_score[item_index]
        else:
            sentence_item = [sentence]
            Print_Item[newID] = sentence_item
            sentence_score[sentence] = title_sentence_score[item_index]
        item_index += 1
    print(f"新闻ID：{len(test)}")
    for i in title_score.keys():
        if i not in test:
            print(i)
        # 从这里开始修改
    for newsId in Print_Item.keys():
        emotion_scores = []
        for sentence in Print_Item[newsId]:
            emotion_scores.append(sentence_score[sentence])

        item = {
            'newsId': newsId,
            'sentences': [newID_title[newsId]] + Print_Item[newsId],
            'emotion_scores': [title_score[newsId]] + emotion_scores
        }
        json_items.append(item)
    # 改到这里
    write_json_data(path_result, json_items)
    title_sentence_file_object.close()
    title_emotion_file_object.close()
    title_sentence_emotion_file_object.close()


root_path = "C:/Users/Jamming/Desktop/souhu/coreEntityEmotion_test_stage2/"
test_stage2 = "test_stage2.txt"
bert_emotion_result = "test_results.tsv"
output_file = "commit_emotion_stage2.tsv"
output_flie_json = root_path+"8w_bert_test_result_json.txt"
# ID_list = get_test_id(test_stage2_path=test_stage2)
# score_list = get_emotion_result(bert_emotion_result)
# print(f"ID_list大小：{len(ID_list)}")
# print(f"score_list大小：{len(score_list)}")
# print_commit_reslut(output_file, ID_list, score_list)


# item_list = get_emotion_result_value(bert_emotion_result)
# print(len(item_list))
# print(len(ID_list))
# bert_result_to_json(output_flie_json, ID_list, item_list)
root_path_2 = "C:/Users/Jamming/Desktop/souhu/HGongC/candidate_corener_emotion/data/emotion"
title_emotion_result = root_path_2+"/"+"all_title_emotion_results.tsv"
title_sentence_emotion_result = root_path_2+"/"+"/title_sentence_emotion_results.tsv"
title_sentence_test_file = root_path_2+"/"+"title_sentence_data.tsv"
train_path = "C:/Users/Jamming/Desktop/souhu/sohu_data/data/data_v2/coreEntityEmotion_train.txt"
path_result = root_path_2+"/"+"train_title_sentence_bert_result.json"
# trainID = load_trainID(train_path)
# title_sentence_bert_result_json(path_result, trainID, title_emotion_result, title_sentence_emotion_result, title_sentence_test_file)

# 处理train文件的情感得分
root = "C:/Users/Jamming/Desktop/souhu/HGongC/candidate_corener_emotion/data/emotion/"
train_bert_result = root + "title_content_emotion.tsv"
train_output_flie_json = root + "train_bert_result_json.txt"
# item_list = get_emotion_result_value(train_bert_result)
# bert_result_to_json(train_output_flie_json, trainID, item_list)

# 处理8Wtitle以及前20个句子的得分为json格式
stage2_path_result = "C:/Users/Jamming/Desktop/souhu/coreEntityEmotion_test_stage2/coreEntityEmotion_test_stage2.txt"
title_emotion_result = root+"test_stage2_title_results.tsv"
bert_emotion_result_stage2 = root+"8W_stage2_title_sentences_bert_emotion.json"
stage2_title_sentence_emotion_result = root + "8W_title_sentence_test_results.tsv"
stage2_title_sentence_test_file = root + "8W_title_sentence_test_stage2.txt"
test_stage2_ID = load_trainID(stage2_path_result)
title_sentence_bert_result_json(bert_emotion_result_stage2, test_stage2_ID, title_emotion_result, stage2_title_sentence_emotion_result, stage2_title_sentence_test_file)
