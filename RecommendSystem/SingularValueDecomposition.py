# coding=utf-8

from numpy import linalg as la
from numpy import mat

# 奇异值分解
# A = mat([[3, 4, 3, 1], [1, 3, 2, 6], [2, 4, 1, 5], [3, 3, 5, 2]])
# U, sigma, VT = la.svd(A)
# print('U\n', U)
# print('sigma\n', sigma)
# print('Vt\n', VT)


def loadRetrievalContent(filepath):
    newslist = []
    f = open(filepath, 'r',encoding= 'utf8')
    for line in f:
        # line = re.sub(r, '', line)
        print(line)
    f.close()

filename = 'C:\\Users\\Jamming\\Desktop\\trec56_collection_zysx.txt'
loadRetrievalContent(filename)