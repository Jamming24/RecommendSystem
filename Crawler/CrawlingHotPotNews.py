# coding=utf-8

# 爬去知微事见的热点新闻
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

titlelist = []


def parse_html(soup, f):
    info = ""
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
        info = divs_index[index].string+',' + title+',' + link+',' + date+',' + degree+',' + '\n'
        titlelist.append(info)
        f.write(info.replace(u'\xa0', u' '))
        f.flush()


def gethotitle(page, outflie):
    f = open(outflie, 'w')
    browser = webdriver.Firefox()
    browser.maximize_window()
    browser.get('http://ef.zhiweidata.com/#!/down')
    time.sleep(1)

    for i in range(1, page+1):
        soup = BeautifulSoup(browser.page_source, 'lxml')
        # 解析html，获取文章列表
        parse_html(soup, f)
        browser.find_element(By.LINK_TEXT, '>').click()
        print('第', i, '页完成')
        time.sleep(1)
    f.close()
    browser.close()
    return titlelist

