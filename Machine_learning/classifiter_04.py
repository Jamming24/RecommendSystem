# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 13:20
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : classifiter_04.py
# @Software: PyCharm
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import numpy as np

X = 2 * np.random.rand(100, 1)
X_test = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
# 梯度下降
# 学习率
eta = 0.1
n_iterations = 1000
m = 100
# 随机初始化参数
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
print(theta)
'''
1.在使用随机梯度下降算法求解模型参数的时候，可以使用StandardScaler类，对特征值进行归一化处理，可以加快求解速度，使损失函数快速收敛
2.随机梯度算法中的随机性优点是可以逃离局部最优， 但是缺点是永远定位不出最小值
3.我们使用交叉验证来评估模型的泛化性能，如果模型在训练集上表现良好，但是在交叉验证中表现非常糟糕，那就是过度拟合，如果在训练和交叉验证集合上表现的都不好，
 那就是拟合不足
4.方差和偏差的权衡 模型的泛化误差可以被表示为三个截然不同的误差之和
 偏差
    导致的原因在于错误的假设，比如假设数据是线性的，其实二次的，高偏差模型最有可能对训练数据的拟合不足
 方差
    误差是由于模型对训练数据的微笑变化过度敏感导致的， 高阶模型也可能有高方差，所以很容易对训练数据造成过拟合
 不可避免的误差
    因为数据噪声所导致的，减少误差的唯一方法就是清洗数据。
5.对于线性回归来说，可以使用alpha控制其正则化程度，对于逻辑回归正好想反，用C控制其正则化，C时alpha的逆反，C越大 正则化程度越高
'''
# 随机梯度下降
n_epochs = 50
t0, t1 = 5, 50
theta = np.random.randn(2, 1)


def learning_schedule(t):
    return t0 / (t + t1)


for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta * gradients
print(theta)
'''
早期停止法的基本实现 当warm_start=True时，调用fit方法，会从停下来的地方开始训练，而不是重新开始
'''
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
minmum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X, y)
    y_pre = sgd_reg.predict(X_test)
    val_error = mean_squared_error(y_pre, y)
    if val_error < minmum_val_error:
        minmum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
print(best_epoch)
print(best_model)
