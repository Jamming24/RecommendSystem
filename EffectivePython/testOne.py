# coding=utf-8

import sys
from urllib.parse import parse_qs
import this

my_values = parse_qs('red=5&blue=0&green=', keep_blank_values=True)
print(my_values)
print(repr(my_values))
test = dict()
if not test:
    print('空字典')
else:
    print(my_values)


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value


def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value


a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
print(a[2:-1])
# 倒数第三个  到  倒数第二个
print(a[-3:-1])
print(a[:-1])
print(to_str('100200033'))
print(to_bytes('1100000'))
# 打印Python版本信息
print(sys.version)
print(sys.version_info)
