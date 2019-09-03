# -*- coding: utf-8 -*-
# @Time    : 2019/6/15 20:18
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : data_process.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import data
import gensim
from rake_nltk import Rake
from nltk.corpus import PlaintextCorpusReader
import warnings
warnings.filterwarnings("ignore")
# 添加自己的NLTK下载数据路径
data.path.append(r"D:\PycharmWorkSpace\Machine_learning\datasets\nltk_data")
# nltk.download('wordnet')

root_path = "E:/Fire2019/评测任务三/"


def standard_generating_data(data_path):
    # 总数：1306118，1225308个0 占比0.93812；80810个1 占比 0.06187  0是正常的 1 是虚假问题
    all_data = pd.read_csv(open(data_path, encoding='UTF-8'))
    print(all_data.shape)
    # print(all_data.describe())
    print(all_data.info())
    print(all_data["target"].value_counts())
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_dev_index, test_index in split.split(all_data, all_data["target"]):
        train_dev_set = all_data.loc[train_dev_index]
        test_set = all_data.loc[test_index]
    train_dev_df = train_dev_set.reset_index().drop("index", axis=1)
    train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, dev_index in train_dev_split.split(train_dev_df, train_dev_df["target"]):
        train_set = train_dev_df.loc[train_index]
        dev_set = train_dev_df.loc[dev_index]
    train_set.to_csv(root_path+"实验文件/train.tsv", sep="\t", index=False, encoding="UTF-8")
    dev_set.to_csv(root_path+"实验文件/dev.tsv", sep="\t", index=False, encoding="UTF-8")
    test_set.to_csv(root_path+"实验文件/test.tsv", sep="\t", index=False, encoding="UTF-8")
    print(f"二分类训练集数据：{train_set.shape[0]}行")
    print(f"二分类开发集数据：{dev_set.shape[0]}行")
    print(f"二分类测试集数据：{test_set.shape[0]}行")

    min_all_data = all_data.loc[0:13061]
    min_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for min_train_dev_index, min_test_index in min_split.split(min_all_data, min_all_data["target"]):
        min_train_dev_set = min_all_data.loc[min_train_dev_index]
        min_test_set = min_all_data.loc[min_test_index]
    min_train_dev_df = min_train_dev_set.reset_index().drop("index", axis=1)
    min_train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for min_train_index, min_dev_index in min_train_dev_split.split(min_train_dev_df, min_train_dev_df["target"]):
        min_train_set = min_train_dev_df.loc[min_train_index]
        min_dev_set = min_train_dev_df.loc[min_dev_index]
    min_train_set.to_csv(root_path+"实验文件/mini_train.tsv", sep="\t", index=False, encoding="UTF-8")
    min_dev_set.to_csv(root_path+"实验文件/mini_dev.tsv", sep="\t", index=False, encoding="UTF-8")
    min_test_set.to_csv(root_path+"实验文件/mini_test.tsv", sep="\t", index=False, encoding="UTF-8")
    print(f"二分类mini训练集数据：{min_train_set.shape[0]}行")
    print(f"二分类mini开发集数据：{min_dev_set.shape[0]}行")
    print(f"二分类mini测试集数据：{min_test_set.shape[0]}行")


def get_ICQtrian_content(id_file, out_file):
    train_ICQ_df = pd.DataFrame()
    csv_data_all = pd.read_csv(open(root_path + '源文件/train.csv', encoding='UTF-8'))
    csv_data_all.drop(["target"], axis=1, inplace=True)
    # ICQ_csv = pd.read_csv(open(root_path+"源文件/CIQ-traindata.tsv", encoding='UTF-8'), sep='\t')
    ICQ_csv = pd.read_csv(open(id_file, encoding='UTF-8'), sep='\t')
    # 此时的csv_id 是一个Series 可以通过append方法添加到DateFrame中
    csv_id = ICQ_csv["qid"]
    # csv_target = ICQ_csv["Label"].tolist()
    print(f"六分类数据总计有{csv_id.count()}行")
    for i in range(csv_id.count()):
        df = csv_data_all[csv_data_all['qid'] == csv_id[i]]
        train_ICQ_df = pd.concat([train_ICQ_df, df])
    print("处理完成")
    print(train_ICQ_df.info())
    # train_ICQ_df.insert(2, 'target', csv_target)
    print(train_ICQ_df.info())
    # train_ICQ_df.to_csv(root_path+"实验文件/CIQ_traindata_contents.tsv", sep='\t', index=False, encoding='UTF-8')
    train_ICQ_df.to_csv(out_file, sep='\t', index=False, encoding='UTF-8')
    print("写入成功")


def standard_generating_CIQ_data():
    # 切分出开发集合
    CIQ_data_csv = pd.read_csv(open(root_path+"实验文件/CIQ_traindata_contents.tsv", encoding="UTF-8"), sep="\t")
    print(CIQ_data_csv.info())
    CIQ_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for CIQ_train_dev_index, CIQ_test_index in CIQ_split.split(CIQ_data_csv, CIQ_data_csv["target"]):
        CIQ_test_set = CIQ_data_csv.loc[CIQ_test_index]
        CIQ_train_dev_set = CIQ_data_csv.loc[CIQ_train_dev_index]
    CIQ_test_set.to_csv(root_path+"实验文件/CIQ_test.tsv", encoding='UTF-8', sep='\t', index=False)
    # 重置索引  并且删除旧的索引行
    CIQ_train_dev_csv = CIQ_train_dev_set.reset_index().drop("index", axis=1)
    CIQ_train_dev_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, dev_index in CIQ_train_dev_split.split(CIQ_train_dev_csv, CIQ_train_dev_csv["target"]):
        CIQ_train_set = CIQ_train_dev_csv.loc[train_index]
        CIQ_dev_set = CIQ_train_dev_csv.loc[dev_index]
    CIQ_train_set.to_csv(root_path+"实验文件/CIQ_train.tsv", encoding='UTF-8', index=False, sep='\t')
    CIQ_dev_set.to_csv(root_path+"实验文件/CIQ_dev.tsv", encoding='UTF-8', index=False, sep='\t')
    print(f"CIQ训练集数据：{CIQ_train_set.shape[0]}行")
    print(f"CIQ开发集数据：{CIQ_dev_set.shape[0]}行")
    print(f"CIQ测试集数据：{CIQ_test_set.shape[0]}行")


def main():
    train_data_path = root_path + '源文件/train.csv'
    standard_generating_data(data_path=train_data_path)
    get_ICQtrian_content()
    standard_generating_CIQ_data()


def load_data_contents(data_path):
    csv_data = pd.read_csv(open(data_path, encoding='UTF-8'), sep='\t')
    return csv_data['question_text']


def text_stemmer(contents, flag=1):
    # flag = 1 表示词形还原， 0表示词干提取
    stemmer_contents = []
    lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    for line in contents:
        Stemmer_words = []
        words = word_tokenize(line)
        for w in words:
            if flag == 1:
                stemmer_word = lemmatizer.lemmatize(word=w)
            elif flag == 0:
                stemmer_word = porter_stemmer.stem(word=w)
            Stemmer_words.append(stemmer_word)
        # print("lemmatizer句子", " ".join(Stemmer_words))
        stemmer_contents.append(" ".join(Stemmer_words))
    return pd.Series(stemmer_contents)


def main_2():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    CIQ_train_path = root_path + "实验文件/CIQ_train.tsv"
    data_contents = load_data_contents(CIQ_train_path)
    print(text_stemmer(data_contents, flag=0))


def get_CIQ_wordVector():
    wordlists = PlaintextCorpusReader(root_path + "实验文件", fileids='.*\.tsv')
    words = wordlists.words("CIQ_traindata_contents.tsv")
    model = gensim.models.KeyedVectors.load_word2vec_format(root_path + 'GoogleNews-vectors-negative300.bin',
                                                            binary=True)

    vocabulary = model.wv.vocab.keys()
    file_object = open(root_path + "实验文件/CIQ_wordVector.tsv", encoding='UTF-8', mode='w')
    file_object.write(f"{len(vocabulary)}\t{300}\n")
    for w in words:
        if len(w) != 20 and w in vocabulary:
            vector_list = model.get_vector(w).tolist()
            file_object.write(w + "\t")
            for index in range(len(vector_list) - 1):
                file_object.write(str(vector_list[index]) + ",")
            file_object.write(str(vector_list[len(vector_list) - 1]) + "\n")
    file_object.close()


def get_rake():
    stopwordlist = "E:/Fire2019/评测任务一/AILA-data/stopwords.txt"
    rake = Rake(stopwords=stopwordlist, min_length=2)
    text = "What is the average IQ of Christians?"
    rake.extract_keywords_from_text(text)
    print(rake.rank_list)
    b = rake.get_ranked_phrases()
    print(b)

    # print("hello")
    #
    # r = Rake(min_length=1, max_length=1)
    # my_test = 'My father was a self-taught mandolin player. He was one of the best string instrument players in our town. He could not read music, but if he heard a tune a few times, he could play it. When he was younger, he was a member of a small country music band. They would play at local dances and on a few occasions would play for the local radio station. He often told us how he had auditioned and earned a position in a band that featured Patsy Cline as their lead singer. He told the family that after he was hired he never went back. Dad was a very religious man. He stated that there was a lot of drinking and cursing the day of his audition and he did not want to be around that type of environment.'
    # r.extract_keywords_from_text(my_test)
    # print(r.get_ranked_phrases())
    # print("==============================")
    # print(r.get_ranked_phrases_with_scores())
    # print("===========================")
    # print(r.stopwords)
    # print("=============================")
    # print(r.get_word_degrees())


def get_Insincere_question():
    train_data_path = root_path + "源文件/train.csv"
    train_csv = pd.read_csv(open(train_data_path, encoding='UTF-8'))
    insincere_question = train_csv[train_csv["target"] == 1]
    insincere_question.to_csv(root_path+"源文件/insincere_question.csv", index=False)


# get_ICQtrian_content(id_file=root_path+"源文件/CIQ-testdata_noLabel.tsv", out_file=root_path+"实验文件/CIQ_test_no_Label.tsv")
get_Insincere_question()


