# -*- coding:utf-8 -*-

import json
import requests
import time


class HtmlDownloader(object):
    def download(self, url):
        if url is None:
            return None
        # user_agent = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
        user_agent = "Mozilla/5.0 (compatible; MSIE 5.5; Windows NT)"
        headers = {'User-Agent': user_agent}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.text
        return None


t = time.time()
TouTiaoUrl = "https://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1"
max_behot_time = int(t)
max_behot_time_tmp = int(t)
url = TouTiaoUrl + "&max_behot_time="+str(max_behot_time)+"&max_behot_time_tmp="+str(max_behot_time_tmp)+"&tadrequire=true&as=A105DB7057D6BD2&cp=5B07460BFD123E1&_signature=mj2tLQAAwUYT3JS3cTcO5Zo9rT"
url2 = "https://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1&max_behot_time=0&max_behot_time_tmp=0&tadrequire=true&as=A1E58B00F756852&cp=5B07766815727E1&_signature=5v2YxAAAvgZvHKFer8vkJ-b9mN"

print(int(t))

html = HtmlDownloader()
jsonData = html.download(url)
data = json.loads(jsonData)
newsData = data['data']
for d in newsData:
    print('title:', d['title'])
    print('abstract:', d['abstract'])
    print('behot_time:', d['behot_time'])



