# encoding:utf-8

import string
import csv
import docx2txt
import os
import nltk
import time
from nltk.tokenize import word_tokenize
from win32com import client as wc


def staticalWordCore(doc_content):
    # 统计词频核心部件
    doc_content = doc_content.replace("\n", " ")
    del_str = string.punctuation + string.digits
    replace = " " * len(del_str)
    tran_tab = str.maketrans(del_str, replace)
    # Word_frequence_dict = {}
    sentence = doc_content.translate(tran_tab)
    # word_list = sentence.split(" ")
    word_list = word_tokenize(sentence)
    # while '' in word_list:
    #     word_list.remove('')
    # for word in word_list:
    #     if word in Word_frequence_dict.keys():
    #         num = Word_frequence_dict[word]
    #         Word_frequence_dict[word] = num + 1
    #     else:
    #         Word_frequence_dict[word] = 1
    # word_list.clear()
    # del word_list
    Word_frequence_dict = nltk.FreqDist(word_list)
    return Word_frequence_dict


def read_docx(docx_path, word_type):
    # 支持doc,docx
    flag = 0
    if word_type == "doc" or word_type == "wps":
        flag = 1
        w = wc.Dispatch('Word.Application')
        # 或者使用下面的方法，使用启动独立的进程：
        # w = wc.DispatchEx('Word.Application')
        doc = w.Documents.Open(docx_path)
        docx_path = "E:\\temp_01.docx"
        doc.SaveAs(docx_path, 16)  # 必须有参数16，否则会出错
        doc.Close()
        w.Quit()
    doc_content = docx2txt.process(docx_path)
    Word_frequence_dict = staticalWordCore(doc_content)
    if flag == 1:
        os.remove(docx_path)
    return Word_frequence_dict


def read_csv(csv_path):
    # 支持CSV文件
    doc_content = ""
    csv_reader = csv.reader(open(csv_path))
    for line in csv_reader:
        for content in line:
            doc_content += content+" "
    Word_frequence_dict = staticalWordCore(doc_content)
    del doc_content
    return Word_frequence_dict


def read_text(text):
    doc_content = ""
    file_object = open(text, 'r', encoding='UTF-8')
    for line in file_object:
        doc_content += line
    Word_frequence_dict = staticalWordCore(doc_content)
    file_object.close()
    del doc_content
    return Word_frequence_dict

# def read_wps(wps_file):


def Read_MultiFormat_File(file_path):
    # 支持格式 csv，doc，docx，pdf，json，text，xlsx，zip，xml，html，wps

    num = file_path.rfind('.')
    file_type = file_path[num+1:len(file_path)]
    print(f"文档类型:{file_type}")
    if file_type == 'csv':
        Word_frequence_dict = read_csv(file_path)
    elif file_type == 'txt':
        Word_frequence_dict = read_text(file_path)
    elif file_type == "docx" or file_type == "doc" or file_type == "wps":
        Word_frequence_dict = read_docx(file_path, file_type)
    return Word_frequence_dict


if __name__ == '__main__':
    # root_floder = "E:\\分词实验检查"
    # out_floder = "E:\\第二次分词实验结果"
    # all_file = os.listdir(root_floder)
    # print(all_file)
    # for testfile in all_file:
    #     start_time = time.time()
    #     simply_test = os.path.join(root_floder, testfile)
    #     Word_frequence_dict = Read_MultiFormat_File(simply_test)
    #     already_sort = sorted(Word_frequence_dict.items(), key=lambda e: e[1], reverse=True)
    #     file_object = open(os.path.join(out_floder, testfile+".result"), 'w', encoding='UTF-8')
    #     print(f"{testfile}的无重复词数:{str(len(Word_frequence_dict))}个")
    #     for key in already_sort:
    #         file_object.write(key[0]+":"+str(key[1])+"\n")
    #     file_object.close()
    #     end_time = time.time()
    #     print(f"{testfile}总计用时{use_time}秒")

    simply_test = "E:\\分词实验检查\\20M.doc"
    file_object = open("E:\\第二次分词实验结果\\20M.result", 'w', encoding='UTF-8')
    start_time = time.time()
    Word_frequence_dict = Read_MultiFormat_File(simply_test)
    already_sort = sorted(Word_frequence_dict.items(), key=lambda e: e[1], reverse=True)
    end_time = time.time()
    use_time = end_time-start_time
    for key in already_sort:
        # print(key)
        file_object.write(key[0]+":"+str(key[1])+"\n")
    print(f"总计用时{use_time}秒")
    file_object.close()

