# encoding:utf-8
# 直接计算与test中的问题Id最相关并且在候选集合中的答案ID 可以减少计算量
# 首先 加载test中的所有问题ID 计算与每个Id最相关的100问题（并且只保留在候选集合中的） 返回一个集合
# 遍历test 对于每条用户行为 按照时间从高到排序，最近发生的放到前面，取与每个行为最相关的100条，存到字典中
# 作为推荐结果
#
import numpy.linalg as li
import numpy as np

# 行向量
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
# 列向量
C = np.array([[1, 2, 3]])
num = np.dot(A, B)
num2 = A.T * B
denom = li.norm(A) * li.norm(B)
cos = num / denom  # 余弦值
sim = 0.5 + 0.5 * cos  # 归一化
print("num:"+str(num))
print("num2:"+str(num2))
print(li.norm(A))
print(li.norm(B))
print("denom:"+str(denom))
print("cos:"+str(cos))
print("sim:"+str(sim))
sum = 0
for i in range(0, 420000):
    sum += i
print(sum)



# num:32
# num2:[ 4 10 18]
# 3.7416573867739413
# 8.774964387392123
# denom:32.7998
# cos:0.9756157049738109
# sim:0.9878078524869054

# num:32
# num2:[ 4 10 18]
# 3.7416573867739413
# 8.774964387392123
# denom:32.83291031876401
# cos:0.9746318461970762
# sim:0.9873159230985381