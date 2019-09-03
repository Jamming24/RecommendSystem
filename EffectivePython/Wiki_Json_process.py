# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 9:47
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Wiki_Json_process.py
# @Software: PyCharm

import json
import codecs
from tqdm import tqdm
import thulac

"""
本程序功能是将json形式的维基百科数据进一步进行简繁体转换和拆句处理
"""


def load_json(wiki_json):
    print('load file', wiki_json)
    text_list = []
    error_line = []
    # 使用编码方式读取文件
    f = codecs.open(wiki_json, 'r', encoding='UTF-8')
    # tqdm 是一个可扩展的进度条
    for line in tqdm(f.readlines()):
        try:
            entity = json.loads(line.strip())
            text = entity['text'].replace('\n', '')
            # id = entity['id']
            # url = entity['url']
            # title = entity['title']
            text_list.append(text)
        except:
            error_line.append(line)
            continue
    print(len(error_line))
    # 丢弃了121行异常数据
    return text_list


def Split_Sentence(text_list, sentence_wiki):
    # 将文章拆分为带有标点符号的句子
    file = open(sentence_wiki, 'w', encoding='UTF-8')
    writer_line = []
    for line in text_list:
        sentences = line.split("。")
        for ss in sentences:
            if len(ss) > 0:
                writer_line.append(ss)
    print(f"总计有{len(writer_line)}行数据")
    for w in tqdm(writer_line):
        file.write(w+'\n')
    file.close()


def Segmentation(wiki_file, Words_dict, output_text):
    thu1 = thulac.thulac(user_dict=Words_dict, seg_only=True)
    input_object = open(wiki_file, 'r', encoding="UTF-8").readlines()
    output_object = open(output_text, 'w', encoding="UTF-8")
    for line in tqdm(input_object):
        str = line.replace(" ", "")
        t = thu1.cut(str, text=True)
        output_object.write(t + "\n")
    input_object.close()
    output_object.close()
    print("清华分词处理完成")


root_qi = "E:/Teacher_Qi/"
wiki_json = root_qi + "ZhWiki_extractor/zh_wiki_data.txt"
sentence_wiki = root_qi + "zh_wiki_sentence_data.txt"
# text_list = load_json(wiki_json)
# Split_Sentence(text_list, sentence_wiki)

simplified_wiki = root_qi + "zh_wiki_sentence_simplified.txt"
Segment_output_text = root_qi + "zh_wiki_sentence_simplified_THU_Segment.txt"
dict_path = root_qi + "Words_dict.txt"
Segmentation(simplified_wiki, dict_path, Segment_output_text)





