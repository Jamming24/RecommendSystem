# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
z = tf.matmul(x, W) + b
MaxOut = tf.reduce_max(z, axis=1, keep_dims=True)
# 设置学习参数
W2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
# 构建模型
pred = tf.nn.softmax(tf.matmul(MaxOut, W2)+b2)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
learning_rate = 0.04
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 200
batch_size = 100
display_step = 1
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
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

