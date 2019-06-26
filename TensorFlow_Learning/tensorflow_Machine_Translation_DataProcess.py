# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 17:02
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : tensorflow_Machine_Translation_DataProcess.py
# @Software: PyCharm
import os
import re
import jieba
import numpy as np
import collections
from sklearn.utils import shuffle
from tensorflow.python.platform import gfile

# 系统字符，创建字典的时候加入
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"


def build_dataset(words, n_words):
    """处理整行的输入到一个数据集, 再字典中加入PAD，GO，EOS，UNK是为了再训练模型时起到辅助标记的作用
        PAD用于在桶机制中为了对齐填充占位， GO是解码输出是的开头标志位，EOS是用来标记输出结果的结尾
        UNK用来代替处理样本时出现自字典中没有的字符
    """
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        # 什么意思？？？为什么这样写
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def basic_tokenizer(sentence):
    _WORD_SPLIT = "([.,!?\"':;)(])"
    _CHWORD_SPLIT = '、|。|，|‘|’‘'
    str1 = ""
    for i in re.split(_CHWORD_SPLIT, sentence):
        str1 = str1 + i
    str2 = ""
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2 + i
    return str2


# 获取文件列表
def getRawFileList(path):
    files = []
    names = []
    for file in os.listdir(path=path):
        # file.endswith("~")用了判断字段串是否以指定后缀结尾
        # 用来过滤其他文件的
        if not file.endswith("~") or not file == "":
            files.append(os.path.join(path, file))
            names.append(file)
    return files, names


# 读取分词后的中文词
def get_ch_label(txt_file, Isch=True, normalize_digits=False):
    # txt_file = ".\\data\\Translation_Chinese\\Corpus\\from\\english1w.txt"
    labels = list()
    labelssz = []
    with open(txt_file, 'rb') as f:
        for label in f:
            linstr1 = label.decode('utf-8')
            if normalize_digits:
                linstr1 = re.sub('\d+', _NUM, linstr1)
            notoken = basic_tokenizer(linstr1)

            if Isch:
                notoken = Segment_word(notoken)
            else:
                notoken = notoken.split()
            labels.extend(notoken)
            labelssz.append(len(labels))
    return labels, labelssz


# 获取文件中的文本
def get_ch_path_text(raw_data_dir, Isch=True, normalize_digits=False):
    # getRawFileList() 返回文件路径 和文件名称列表
    text_files, _ = getRawFileList(raw_data_dir)
    labels = []
    training_dataszs = list([0])
    if len(text_files) == 0:
        print("错误，没有文件在文件夹", raw_data_dir)
        return labels, training_dataszs
    print(len(text_files), "个文件, 首个文件是：", text_files[0])
    shuffle(text_files)
    for file in text_files:
        # 获取分词后的中文词作为训练数据
        training_data, training_datasz = get_ch_label(txt_file=file, Isch=Isch, normalize_digits=normalize_digits)
        # ################观察训练数据########################
        # print(training_data, training_datasz)
        # ##################################################
        training_word = np.array(training_data)
        training_word = np.reshape(training_word, [-1, ])
        labels.append(training_word)

        training_datasz = np.array(training_datasz) + training_dataszs[-1]
        training_dataszs.extend(list(training_datasz))
    return labels, training_dataszs


def Segment_word(training_data):
    # 默认分词为精确模式
    seg_list = jieba.cut(training_data)
    training_word = " ".join(seg_list)
    training_word = training_word.split()
    return training_word


# Isch = true 中文， false 英文
# 创建字典，max_vocabulary_size = 500 代表字典有500个词
def create_vocabulary(vocabulary_file, raw_data_dir, max_vocabulary_size, Isch=True, normalize_diglts=False):
    texts, textssz = get_ch_path_text(raw_data_dir=raw_data_dir, Isch=Isch, normalize_digits=normalize_diglts)
    # print("行数：", len(textssz), textssz)
    # 处理多行文本texts
    all_words = []
    for label in texts:
        print("词数:", len(label))
        # ????
        all_words += [word for word in label]
    print("词数:", len(all_words))
    # 调用build_dataset()函数
    training_label, count, dictionary, reverse_dictionary = build_dataset(words=all_words, n_words=max_vocabulary_size)
    # print("reverse_dictionary", reverse_dictionary, len(reverse_dictionary))
    if not gfile.Exists(vocabulary_file):
        print("Creating vocabulary %s from data %s" % (vocabulary_file, data_dir))
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
        with gfile.GFile(vocabulary_file, mode="w") as vocab_file:
            for w in reverse_dictionary:
                # print(reverse_dictionary[w])
                vocab_file.write(str(reverse_dictionary[w]) + "\n")
    else:
        print("词表已经存在，什么都不需要做了")
    return training_label, count, dictionary, reverse_dictionary, textssz


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# 将文件批量转换为ids文件
def textdir_to_idsdir(textdir, idsdir, vocab, normalize_digits=True, Isch=True):
    text_files, filenames = getRawFileList(textdir)
    # np.reshape(training_dataszs,(1,-1))
    if len(text_files) == 0:
        raise ValueError("err:no files in ", raw_data_dir)

    print(len(text_files), "files,one is", text_files[0])

    for text_file, name in zip(text_files, filenames):
        textfile_to_idsfile(text_file, idsdir + name, vocab, normalize_digits, Isch)


# 将一个文件转成ids 不是windows下的要改编码格式 utf8
def textfile_to_idsfile(data_file_name, target_file_name, vocab, normalize_digits=True, Isch=True):
    if not gfile.Exists(target_file_name):
        print("Tokenizing data in %s" % data_file_name)
        with gfile.GFile(data_file_name, mode="rb") as data_file:
            with gfile.GFile(target_file_name, mode="w") as ids_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    # token_ids = sentence_to_ids(line.decode('gb2312'), vocab,normalize_digits,Isch)
                    token_ids = sentence_to_ids(line.decode('utf8'), vocab, normalize_digits, Isch)
                    ids_file.write(str(" ".join([str(tok) for tok in token_ids])) + "\n")
                    # print(" ".join([str(tok) for tok in token_ids]))


#将句子转成ids
def sentence_to_ids(sentence, vocabulary, normalize_digits=True, Isch=True):
    if normalize_digits:
        sentence=re.sub('\d+', _NUM, sentence)
    notoken = basic_tokenizer(sentence)
    if Isch:
        notoken = Segment_word(notoken)
    else:
        notoken = notoken.split()
    idsdata = [vocabulary.get(w) for w in notoken]
    return idsdata


# 将ID转换为文本
def ids2texts(indices, rev_vocab):
    texts = []
    for index in indices:
        #texts.append(rev_vocab[index].decode('ascii'))
        texts.append(rev_vocab[index])
    return texts

plot_histograms = plot_scatter = True
vocab_size = 400000

max_num_lines = 1
max_target_size = 200
max_source_size = 200

data_dir = ".\\data\\Translation_Chinese\\"
raw_data_dir = ".\\data\\Translation_Chinese\\Corpus\\from"
raw_data_dir_to = ".\\data\\Translation_Chinese\\Corpus\\to"
vocabulary_file_en = "en_dict.txt"
vocabulary_file_ch = "ch_dict.txt"

jieba.load_userdict(".\\data\\myjiebadict.txt")

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# 文字字符替换，不属于系统字符
_NUM = "_NUM"


def main():
    vocabulary_fileName_en = os.path.join(data_dir, vocabulary_file_en)
    vocabulary_fileName_ch = os.path.join(data_dir, vocabulary_file_ch)
    # 创建英文字典
    training_data_en, count_en, dictionary_en, reverse_dictionary_en, textsss_en = \
        create_vocabulary(vocabulary_file=vocabulary_fileName_en, raw_data_dir=raw_data_dir, max_vocabulary_size=vocab_size,
                          Isch=False, normalize_diglts=True)
    print("training_data", len(training_data_en))
    print("dictionary", len(dictionary_en))
    # 创建中文字典
    training_data_ch, count_ch, dictionary_ch, reverse_dictionary_ch, textsss_ch = \
        create_vocabulary(vocabulary_file=vocabulary_fileName_ch, raw_data_dir=raw_data_dir_to, max_vocabulary_size=vocab_size,
                          Isch=True, normalize_diglts=True)
    print("training_data_ch", len(training_data_ch))
    print("dictionary_ch", len(dictionary_ch))

    vocaben, rev_vocaben = initialize_vocabulary(vocabulary_fileName_en)
    vocabch, rev_vocabch = initialize_vocabulary(vocabulary_fileName_ch)
    textdir_to_idsdir(raw_data_dir, data_dir+"fromids\\", vocaben, normalize_digits=True, Isch=False)
    textdir_to_idsdir(raw_data_dir_to, data_dir + "toids\\", vocabch, normalize_digits=True, Isch=True)


# main()
