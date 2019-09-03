# -*- coding:utf-8 -*-

from bert_serving.client import BertClient

root_path = "E:/Teacher_Qi/"


def load_metric_words():
    word_sim297 = open(root_path+"297.txt", 'r', encoding='UTF-8')
    word_sim240 = open(root_path+"240.txt", 'r', encoding='UTF-8')
    words = set()
    for line in word_sim297:
        t = line.split("\t")
        words.add(t[0].strip())
        words.add(t[1].strip())
    for line in word_sim240:
        t = line.split("\t")
        words.add(t[0].strip())
        words.add(t[1].strip())
    print(f"词表总数{len(words)}")
    return words


def get_Vector(out_path):
    words = list(load_metric_words())
    bc = BertClient(check_version=False, check_length=False, port=5555, port_out=5556)
    vec = bc.encode(words)
    print(f"获得词向量总数{len(vec)}")
    vector_file = open(out_path, 'w', encoding='UTF-8')
    size = len(vec[0])
    print(f"词向量维数{size}")
    vector_file.write(str(len(words)) + " " + str(size) + "\n")
    for index in range(len(words)):
        vector_file.write(words[index] + " ")
        for weight in vec[index]:
            vector_file.write(str(weight) + " ")
        vector_file.write("\n")
    vector_file.close()


bert_vector = root_path + "chinese_bert_vector.txt"
get_Vector(bert_vector)
# 词表总数563
# 获得词向量总数563
# 词向量维数768
# bc = BertClient(check_version=False, check_length=False, port=5555, port_out=5556)
# vec = bc.encode(['以色列'])
# print(vec)