# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 19:51
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : CCMT_process_data.py
# @Software: PyCharm

root_path = 'E:/CCMT/数据/'
parallel_data = '1.汉英新闻领域机器翻译-CCMT2019-CE-HLJIT/parallel/CASICT2015/casict2015/'
chinese_file = "casict2015_ch.txt"
english_file = "casict2015_en.txt"


def load_translate_data(chinese_file, english_file):
    docs_source = []
    docs_target = []
    ch_data = open(root_path+parallel_data+chinese_file, 'r', encoding='UTF-8')
    en_data = open(root_path+parallel_data+english_file, 'r', encoding='UTF-8')
    for line in ch_data:
        print(line)

    for line in en_data:
        print(line)
    ch_data.close()
    en_data.close()
    # docs_source.append(doc_source)
    # docs_target.append(doc_target)


load_translate_data(chinese_file, english_file)
