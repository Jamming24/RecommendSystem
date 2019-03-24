# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 生成one-hot编码
def onehot(y, start, end):
    ohe = OneHotEncoder()
    ohe.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    # 这个fit中，所有的数组第一个元素取值分别为：0，1，0，1（黄色标注的），最大为1，且为两种元素（0，1），
    # 说明用2个状态位来表示就可以了，且该维度的value值为2（该值只与最大值有关系，最大值为1）
    # 所有的数组第二个元素取值分别为：0，1，2，0（红色标注的），最大为2，且为两种元素（0，1，2），
    # 说明用3个状态位来表示就可以了，且该维度的value值为3（该值只与最大值有关系，最大值为2）
    # 所有的数组第三个元素取值分别为：3，0，1，2（天蓝色标注的），最大为3，且为两种元素（0，1，2，3），
    # 说明用4个状态位来表示就可以了，且该维度的value值为4（该值只与最大值有关系，最大值为4）
    # 所以整个的value值为（2，3，4），这也就解释了enc.n_values_等于array([2, 3, 4])的原因
    print(ohe.n_values_)
    # enc.transform就是将[0, 1, 1]这组特征转换成onehot编码，toarray()则是转成数组形式。[0, 1, 1],
    # 第一个元素是0，由于之前的fit的第一个维度为2（有两种表示：10，01.程序中10表示0，01表示1），所以用1，0
    # 表示用黄色标注）；第二个元素是1，由于之前的fit的第二个维度为3（有三种表示：100，010，001.
    # 程序中100表示0，010表示1，001表示2），所以用0，1，0表示用红色标注）；第三个元素是1，由于之前的
    # fit的第三个维度为4（有四种表示：1000，0100，0010，0001.程序中1000表示0，0100表示1，0010表示2，0001表示3）
    # 所以用0，1，0，0（用天蓝色标注）表示。综上所述：[0, 1, 1]就被表示为array([[1., 0., 0., 1., 0., 0., 1., 0., 0.]])。
    a = ohe.transform([[0, 1, 1]]).toarray()
    print(a)
    # 转换为onehot编码
    one = OneHotEncoder()
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    one.fit(b)
    c = one.transform(y).toarray()
    return c


# 生成数据函数
def generate(sample_size, num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    # 生成对角矩阵
    cov = np.eye(2)
    samples_per_class = int(sample_size / num_classes)
    X0 = np.random.multivariate_normal(mean=mean, cov=cov, size=samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
    if regression == False:
        # 生成one-hot编码
        # reshape（重塑）用来改变数组形状 a = [1,2,3,4,5,6] a.reshape((2,3)) 就会变成2行3列的数组
        # 注意是-1到1之间
        Y0 = np.reshape(Y0, [-1, 1])
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
    X, Y = shuffle(X0, Y0)
    return X, Y


# 线性多分类
############################
def binary_classes():
    input_dim = 2
    np.random.seed(10)
    num_classes = 2
    mean = np.random.randn(num_classes)
    cov = np.eye(num_classes)
    X, Y = generate(1000, num_classes, [3.0], True)
    # colors = ['r' if l == 0 else 'b' for l in Y[:]]
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.xlabel("Scaled age (in yrs)")
    # plt.ylabel("Tumor size (in cm)")
    # plt.show()

    lab_dim = 1
    input_features = tf.placeholder(tf.float32, [None, input_dim])
    input_labels = tf.placeholder(tf.float32, [None, lab_dim])
    # 定义学习参数
    W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="Weight")
    b = tf.Variable(tf.zeros([lab_dim]), name='bias')

    output = tf.nn.sigmoid(tf.matmul(input_features, W) + b)
    cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
    ser = tf.square(input_labels - output)
    loss = tf.reduce_mean(cross_entropy)
    err = tf.reduce_mean(ser)
    # 收敛速度快，会动态调节梯度
    optimizer = tf.train.AdamOptimizer(0.04)
    train = optimizer.minimize(loss)
    maxEpochs = 50
    minibatchSize = 25
    # 启动session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 向模型输入数据
        for epoch in range(maxEpochs):
            summer = 0
            for i in range(np.int32(len(Y) / minibatchSize)):
                x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
                y1 = np.reshape(Y[i * minibatchSize:(i + 1) * minibatchSize], [-1, 1])
                tf.reshape(y1, [-1, 1])
                _, lossval, outputval, errval = sess.run([train, loss, output, err],
                                                         feed_dict={input_features: x1, input_labels: y1})
                summer = summer + errval
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval),
                  "err=", summer / np.int32(len(Y) / minibatchSize))

        train_X, train_Y = generate(1000, num_classes, [3.0], True)
        colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
        plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)
        # 模型生成的Z用公式可以表示为z=x1*w1+x2*w2+b，将x1和x2映射到x和y坐标
        # 那么z就被分为大于零和小于零两部分。当z=0 表示直线本身。
        # 令上面的公式z等于0 就可以转换为 x2=-x1*w1/w2-b/w2 其中w1，w2，b都是模型学习的参数
        x = np.linspace(-1, 8, 200)
        y = -x * (sess.run(W)[0] / sess.run(W)[1]) - sess.run(b) / sess.run(W)[1]
        plt.plot(x, y, label='Fitted line')
        plt.legend()
        plt.show()


# 线性多分类
# Linear multi-classification
def linear_multi_classification():
    input_dim = 2
    num_classes = 3
    X, Y = generate(2000, num_classes, [[3.0], [3.0, 0]], False)
    aa = [np.argmax(l) for l in Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    # 将具体的点按照不同的颜色显示出来
    # plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.xlabel("Scaled age (in yrs)")
    # plt.ylabel("Tumor size (in cm)")
    # plt.show()
    lab_dim = num_classes
    # 定义占位符
    input_features = tf.placeholder(tf.float32, [None, input_dim])
    input_lables = tf.placeholder(tf.float32, [None, lab_dim])
    # 定义学习参数
    W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="Weight")
    b = tf.Variable(tf.zeros([lab_dim]), name="bias")
    output = tf.matmul(input_features, W) + b
    z = tf.nn.softmax(output)
    # 按照行找出最大索引，生成数组
    a1 = tf.argmax(z, axis=1)
    b1 = tf.argmax(input_lables, axis=1)
    # 两个数组相减，不为零的就是错误个数
    err = tf.count_nonzero(a1 - b1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_lables, logits=output)
    # 对交叉熵取均值
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.04)
    train = optimizer.minimize(loss)

    maxEpochs = 50
    minibatchSize = 25
    # 启动Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(maxEpochs):
            sumerr = 0
            for i in range(np.int32(len(Y) / minibatchSize)):
                x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
                y1 = Y[i * minibatchSize:(i + 1) * minibatchSize, :]
                _, lossval, outputval, errval = sess.run([train, loss, output, err],
                                                         feed_dict={input_features: x1, input_lables: y1})
                sumerr = sumerr + (errval / minibatchSize)
            print("Epoch:", '%04d' % (epoch + 1), "Cost:", "{:.9f}".format(lossval), "err:", sumerr / minibatchSize)

        train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
        aa = [np.argmax(l) for l in train_Y]
        colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
        plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)

        # 画出分类直线
        # x = np.linspace(-1, 8, 200)
        # y = -x * (sess.run(W)[0][0]/sess.run(W)[1][0]) - sess.run(b)[0]/sess.run(W)[1][0]
        # plt.plot(x, y, label='first line', lw=3)
        # y = -x * (sess.run(W)[0][1]/sess.run(W)[1][1])-sess.run(b)[1]/sess.run(W)[1][1]
        # plt.plot(x, y, label='second line', lw=2)
        # y = -x * (sess.run(W)[0][2] / sess.run(W)[1][2]) - sess.run(b)[2] / sess.run(W)[1][2]
        # plt.plot(x, y, label='third line', lw=1)
        #
        # plt.legend()
        # plt.show()
        # print(sess.run(W), sess.run(b))
        # 模型可视化
        nb_of_xs = 200
        xs1 = np.linspace(-1, 8, num=nb_of_xs)
        xs2 = np.linspace(-1, 8, num=nb_of_xs)
        # 创建网格
        xx, yy = np.meshgrid(xs1, xs2)
        # 初始化和填充
        classification_plane = np.zeros((nb_of_xs, nb_of_xs))
        for i in range(nb_of_xs):
            for j in range(nb_of_xs):
                classification_plane[i, j] = sess.run(a1, feed_dict={input_features: [[xx[i, j], yy[i, j]]]})
        # 创建color map用于显示
        cmap = ListedColormap([colorConverter.to_rgba('r', alpha=0.30), colorConverter.to_rgba('b', alpha=0.30), colorConverter.to_rgba('y', alpha=0.30)])
        # 图示各个样本边界
        plt.contourf(xx, yy, classification_plane, cmap=cmap)
        plt.show()


# if __name__ == '__main__':
    # binary_classes()
    # linear_multi_classification()

