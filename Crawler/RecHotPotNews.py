# coding=utf-8

import pymysql
import jieba
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

titlelist = []


# 提取关键词功能模块

def getKeyWords(NewTopicList):
    keysWordlist = []
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for news in NewTopicList:
        line = re.sub(r, '', news.split(',')[1])
        seg_list = jieba.cut(line, cut_all=False)
        keys = '/ '.join(seg_list)
        # print('精确模式Default:', keys)  # 精确模式
        keysWordlist.append(keys)
        # seg_list = jieba.cut(news.split(',')[1], cut_all=True)
        # print('全模式:', '/ '.join(seg_list))  # 全模式
        # seg_list = jieba.cut_for_search(news.split(',')[1])  # 搜索引擎模式
    return keysWordlist


# #############################


# 爬去知微事件上的数据模块

def parse_html(soup, f):
    divs_information = soup.find_all('div', {'class': 'eventListTitle'})
    divs_date_degree = soup.find_all('div', {'class': 'aBox'})
    divs_index = soup.find_all('div', {'class': 'index'})
    for index in range(0, len(divs_information)):
        degree_date = str(divs_date_degree[index])
        date = degree_date[degree_date.find('>') + 1:45].strip()
        degree = degree_date[degree_date.find('>', 46) + 1:degree_date.find('>', 46) + 5]
        link = divs_information[index].a.get('href')
        title = divs_information[index].a.string
        # 获取排名 新闻主题 链接 时间 影响力指数
        info = divs_index[index].string + ',' + title + ',' + link + ',' + date + ',' + degree + ',' + '\n'
        titlelist.append(info)
        f.write(info.replace(u'\xa0', u' '))
        f.flush()


def gethotitle(pagenum, outflie):
    f = open(outflie, 'w')
    browser = webdriver.Firefox()
    browser.maximize_window()
    browser.get('http://ef.zhiweidata.com/#!/down')
    time.sleep(1)

    for i in range(1, pagenum + 1):
        soup = BeautifulSoup(browser.page_source, 'lxml')
        # 解析html，获取文章列表
        parse_html(soup, f)
        browser.find_element(By.LINK_TEXT, '>').click()
        print('第', i, '页完成')
        time.sleep(1)
    f.close()
    browser.close()
    return titlelist


# #####################

# 连接数据库进行模糊查找模块

def getSQl(keys):
    SQLs = []
    for k in keys:
        Sql = "select aid from wh_pre_portal_article_title where title like \"%" + k + "%\" "
        SQLs.append(Sql)
    return SQLs


# 打开数据库连接（ip/数据库用户名/登录密码/数据库名）
def connetDatabase(Host, userName, passwords, DBname):
    DB = pymysql.connect(host=Host, port=3306, user=userName, password=passwords, db=DBname, charset='utf8')
    return DB


def getNewsID(host, user, password, db, page):
    NewsID = []
    connetDatabase(host, user, password, db)
    db = connetDatabase(host, user, password, db)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 每个核心词 都模糊匹配一次，然后结果取并集
    # 热点新闻输出文件路径
    file = 'C:\\Users\\Jamming\\Desktop\\表\\Hotpot_News.txt'
    # 设置获取前p页

    # 获取热点新闻标题列表
    titlelist = gethotitle(page, file)
    # 获取 将热点新闻标题分词后的列表
    keysWordlist = getKeyWords(titlelist)
    print(keysWordlist)
    for keys in keysWordlist:
        unionist = []
        temp = keys.split('/ ')
        # 得到拼接好的SQL语句
        SQLs = getSQl(temp)
        for sql in SQLs:
            # 使用 execute()  方法执行 SQL 查询
            print(sql)
            cursor.execute(sql)
            # 获取剩余结果所有数据
            row = cursor.fetchmany(20)
            unionist = list(set(unionist).union(set(list(row))))
        print('每个热点新闻的ID', unionist)
        NewsID.append(unionist)
        print('输出完成一组>>>>>>>>>')

    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    db.close()
    print(NewsID)
    return NewsID

# ##############################

# host = "60.205.213.252"
# user = "whshop"
# password = "whshop123"
# db = "test_whshop"

host = 'localhost'
user = 'root'
password = '1234'
db = 'test_whshop'
page = 10

getNewsID(host, user, password, db, page)

