# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data


# 使用隐藏层解决非线性问题
def fitting_XOR_with_hidden_layer():
    # 学习速率为0.0001
    learning_rate = 1e-4
    n_input = 2
    n_label = 1
    n_hidden = 2

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_label])
    weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
               'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))}
    biases = {'h1': tf.Variable(tf.zeros([n_hidden])), 'h2': tf.Variable(tf.zeros(n_label))}
    # 定义网络结构
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
    y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['h2']))
    loss = tf.reduce_mean((y_pred) ** 2)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # 生成数据
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('int16')
    # 加载Session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(10000):
        sess.run(train_step, feed_dict={x: X, y: Y})
    # 计算预测值
    print(sess.run(y_pred, feed_dict={x: X}))
    # 查看隐藏层输出
    print(sess.run(layer_1, feed_dict={x: X}))


def MNIST_with_hidden_layer():
    # 利用全连接神经网络进行手写数字识别
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 定义参数
    learning_rate = 1e-3
    training_epochs = 25
    batch_size = 100
    display_step = 1
    # 设置网络模型参数
    n_input = 784
    n_classes = 10
    n_hidden_1 = 256
    n_hidden_2 = 256
    # 定义占位符
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # 学习参数
    weights = {'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
               'h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_2])),
               'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))}
    biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
              'b2': tf.Variable(tf.random_normal([n_hidden_2])),
              'out': tf.Variable(tf.random_normal([n_classes]))}
    # 输出值
    pred = multilayer_perceptron(x, weights, biases)
    # 定义loss和优化器
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 启动循环开始训练
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)
            # 使用随机批梯度下降循环所以数据集
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 运行优化器
                c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                # 计算平均loss值
                # print(type(c))
                # print(c)
                avg_cost += c[1] / total_batch
            # 显示训练中的详细信息
            if (epoch + 1) % display_step == 0:
                # '%04d' % (epoch+1) 表示格式化输出 4表示 总计位数
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Training Finished!")
        # 测试Model  用tf.arg_max返回onehot编码中数值为1那个元素的下标 这三行代码都是干什么用的呢？？？？？？？
        correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    #     # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# 创建model
def multilayer_perceptron(x, weights, biases):
    # 第一层隐藏层
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 输出层
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


if __name__ == '__main__':
    # MNIST_with_hidden_layer()
    # 全连接神经网路中的训练技巧
    # 全连接神经网络容易出现过拟合和欠拟合问题
    # 对于欠拟合问题，可以增加节点或者增加隐藏层的方式，让模型具有更高的拟合性
    # 对于过拟合问题，常用的方法有，early stopping，数据集扩增，正则化，dropout
    # early stopping：在过拟合之前提前结束训练，就是减少训练步数，
    # 数据集扩增：扩大数据集，让模型见到更多的情况
    # 正则化：在损失函数中加入正则化项，如L1正则化，L2正则化
    # dropout：是网络模型中的一种方法，每次训练的时候舍去一些节点来增强泛化能力
    # 对于正则化使用方法：l2正在化在tensorflow有实现，使用方法如下
    reg = 0.01
    weights = {'h1': tf.Variable(tf.random_normal([10, 10])),
               'h2': tf.Variable(tf.random_normal([10, 10]))}
    # 这两个参数 分别对应预测值和真实值
    y_pred = 0
    y = 1
    # 在损失函数加上l2正则化项
    loss = tf.reduce_mean((y_pred-y)**2+tf.nn.l2_loss(weights['h1'])*reg + tf.nn.l2_loss(weights['h2'])*reg)
    # l1正则化项自己实现 tf.reduce_sum(tf.abs(w))
    # 使用dropout来解决过拟合问题
    # 学习速率为0.0001
    learning_rate = 1e-4
    n_input = 2
    n_label = 1
    n_hidden = 2

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_label])
    weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
               'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))}
    biases = {'h1': tf.Variable(tf.zeros([n_hidden])), 'h2': tf.Variable(tf.zeros(n_label))}
    # 定义网络结构
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
    # 使用dropout解决过拟合问题 在layer_1后面加入一个dropout层，将dropout设置为占位符，这样可以随时指定
    # keep_prob, 在run中指定keep_prob为0.6，这意味着每次训练时仅允许0.6的节点参与运算。
    keep_prob = tf.placeholder("float")
    layer_1_drop = tf.nn.dropout(layer_1, keep_prob=keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['h2'])
    # leaky relus激活函数
    y_pred = tf.maximum(layer_2, 0.01*layer_2)

    loss = tf.reduce_mean((y_pred) ** 2)
    # 在训练过程中使用退化学习率
    global_step = tf.Variable(0, trainable=False)
    decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step, decay_steps=10, decay_rate=0.9)

    train_step = tf.train.AdamOptimizer(learning_rate=decaylearning_rate).minimize(loss)
    # 生成数据
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('int16')
    # 加载Session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(10000):
        sess.run(train_step, feed_dict={x: X, y: Y, keep_prob: 0.6})
    # 计算预测值
    print(sess.run(y_pred, feed_dict={x: X}))
    # 查看隐藏层输出
    print(sess.run(layer_1, feed_dict={x: X}))
    # 代码无法运行，仅供举例使用
    # 在搭建网络的时候 需要更深的神经网络，来减少网络中的神经元的数量，使网络具有更好的泛化能力

