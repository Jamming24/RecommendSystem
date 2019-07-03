# -*- coding: utf-8 -*-
# @Time    : 2019/6/29 10:14
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : rake.py
# @Software: PyCharm
from rake_nltk import Rake
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
# ------------------------------
# 孙琳琳同学任务单
# 数据分析， 按照类别分析训练数据， 对每一类别数据首先进行词形还原（词性还原代码我给你），
# 然后统计每次词在这个类别的词频，每个词在数据中的出现的次数（在一条数据中出现算一次），
# 按照词频进行排序，统计

