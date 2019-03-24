# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 激活函数简介
x = tf.constant(3.0)
# 分类函数
# 包括softmax, 和log_softmax 参数随便写的，log_softmax是对softmax取对数
Logits = np.array([12, 3, 2])
tf.nn.softmax(logits=Logits, name=None)
tf.nn.log_softmax(logits=Logits, name=None)


def Swish(X, beta=2):
    return X * tf.nn.sigmoid(x*beta)


with tf.Session() as sess:
    # 激活函数简介
    init = tf.global_variables_initializer()
    sess.run(init)
    # Sigmoid函数
    sigmoid = sess.run(tf.nn.sigmoid(x, name=None))
    print("Sigmoid:", sigmoid)
    # Tanh函数
    tanh = sess.run(tf.nn.tanh(x, name=None))
    print("Tanh:", tanh)
    # Relu函数
    relu = sess.run(tf.nn.relu(x, name=None))
    print("Relu:", relu)
    # Relu6函数 其他还包括Leaky_relu,selu
    relu6 = sess.run(tf.nn.relu6(x, name=None))
    print("Relu6", relu6)
    # softplus 函数
    softplus = sess.run(tf.nn.softplus(x, name=None))
    print("softplus:", softplus)
    leaky_relu = sess.run(tf.nn.leaky_relu(x, alpha=0.2, name=None))
    print("leaky_relu:", leaky_relu)
    # selu
    selu = sess.run(tf.nn.selu(x, name=None))
    print("selu:", selu)
    # Swish函数 tensorflow暂时好像还没有支持swish激活函数 可以自己封装一个
    swish = sess.run(Swish(x, beta=3))
    print("Swish:", swish)

# 关于损失函数 包括均值平方差公式（MSE，Mean squared Error）和交叉熵公式（crossentropy）
# 在tensorflow中没有单独的MSE函数，所以有以下几种实现方式
# 代码中logits表示标签值，outputs表示预测值
Logits = 0.11
outputs = 0.1
MSE = tf.reduce_mean(tf.pow(tf.sub(Logits, outputs), 2.0))
MSE = tf.reduce_mean(tf.square(tf.sub(Logits, outputs)))
MSE = tf.reduce_mean(tf.square(Logits - outputs))
# 常用的交叉熵函数包括，Sigmoid交叉熵，softmax交叉熵，Sparse交叉熵函数，加权Sigmoid交叉熵
targets = []
labels = []
# 计算logits和targets的交叉熵
tf.nn.sigmoid_cross_entropy_with_logits(Logits, targets, name=None)
# 计算logits和targets的交叉熵 Logits和labels必须为相同的shape的数据类型
tf.nn.softmax_cross_entropy_with_logits(Logits, labels, name=None)
# 计算logits和labels的交叉熵，与softmax_cross_entropy_with_logits功能一样，区别在于sparse的样本与真实值不需要one-hot编码
# 但是要求分类的个数一定要从零开始 比如二分类 一定要是零和一这两个数
tf.nn.sparse_softmax_cross_entropy_with_logits(Logits, labels, name=None)
# 在交叉熵的基础上，给第一项乘以一个系数（加权），是增加或减少正样本的损失值
tf.nn.weighted_cross_entropy_with_logits()

