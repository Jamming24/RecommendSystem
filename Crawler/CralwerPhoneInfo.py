# coding=utf-8

import requests
import re
from bs4 import BeautifulSoup


def get_html(url_main):
    """get the content of the url"""
    response = requests.get(url_main)
    response.encoding = 'GBK'
    return response.text


def get_information(html):
    phone_logo = re.search('<h1 class="product-model__name">(.*?)</h1>', get_html(html), flags=0)
    time_to_market = re.search('<span class="section-header-desc">上市时间：(.*?)</span>', get_html(html), flags=0)
    phone_price = re.search('<b class="price-type">(.*?)</b>', get_html(html), flags=0)
    # 返回列表 手机型号, 手机报价, 上市时间
    log = phone_logo.group(1)
    price = phone_price.group(1)
    try:
        time = time_to_market.group(1)
    except AttributeError:
        time = "2018年4月12日"
    return log, price, time


def get_PhoneLogo_link(main_url):
    log_div = re.findall(re.compile('<span class="all active">不限</span>(.*?)</div>'), get_html(main_url))
    log_list = re.findall(re.compile('<a href="(.*?)</a>'), log_div[0])
    # 存储品牌和 品牌链接主页打字典
    Phone_logo = {}
    for li in log_list:
        link = "http://detail.zol.com.cn" + li
        log = link.split('">')
        Phone_logo[log[1]] = log[0]
    return Phone_logo


def get_SimpleLogo_List(logo_url):
    # 本函数已经能够返回某品牌手机型号和报价信息，只需呀传入参数 手机品牌的主页网址即可
    dict = {}
    html_text = get_html(logo_url)
    soup = BeautifulSoup(html_text, 'lxml')
    PhoneList_ul = str(soup.find_all('ul')[3])
    informationlist = BeautifulSoup(PhoneList_ul, 'lxml').find_all('li')
    for i in informationlist:
        info = str(i.get_text())  # 手机的基本信息
        info_list = info.split('\n')
        info_name = info_list[2].split(" ")
        # print('手机名称:', info_name[0], info_name[1])
        # print('手机报价:', info_list[4])
        dict[info_name[0] + info_name[1]] = info_list[4]
    return dict


# 手机列表主页
url = "http://detail.zol.com.cn/cell_phone_index/subcate57_list_1.html"

f = open('C:\\Users\\Jamming\\Desktop\\表\\Phone_information.txt','w')
dict_Phone = get_PhoneLogo_link(url)
for k in dict_Phone:
    print(k)
    dictMap = get_SimpleLogo_List(dict_Phone[k])
    for key in dictMap:
        print(key + '>>>' + dictMap[key])
        f.write(key + ':' + dictMap[key]+'\n')
f.close()
