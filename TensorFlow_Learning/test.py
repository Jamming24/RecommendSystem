# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 21:24
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : test.py
# @Software: PyCharm


def test(a, flag, last_index):
    sublist = a[:last_index]
    if flag == 0:
        sublist.sort()
    elif flag == 1:
        sublist.sort(reverse=True)
    for i in range(len(sublist)):
        a[i] = sublist[i]
    return a


a = [1, 2, 4, 3]
# flag = 0表示升序 1表示降序
# last_index 表示 前last_index个
test(a, flag=1, last_index=3)
test(a, flag=0, last_index=2)
n=100
if n % 2 ==0:
    print(n * n/4)
else:
    print((n*n-1)/4)
