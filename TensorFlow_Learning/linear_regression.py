# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# 保存模型代码

train_X = np.linspace(-1, 1, 100)
# y=2x 但是加入了噪声
# np.random.rand(* train_X.shape) 意思等同于 np.random.randn(100)
train_Y = 2 * train_X + np.random.rand(*train_X.shape) * 0.3
# 显示模拟数据点
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()
tf.reset_default_graph()
# 创建模型
# 占位符  一个代表x的输入, 一个代表对应的真实值y
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数 w被初始化为[-1，1]的随机数，形状为一维的数字，b被初始化为0， 形状也是一维数字 Variable定义变量
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构， multiply是两个数相乘的意思
z = tf.multiply(X, W) + b

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))

# 将损失以标量的形式显示 为了使用tensor board
tf.summary.scalar('loss_function', cost)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 训练模型  tensorflow中的任务是通过session来运行的
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2

# 生成saver
saver = tf.train.Saver()
# 启动session
with tf.Session() as sess:
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    # 合并所有summary
    merged_summary_op = tf.summary.merge_all()
    # 创建summary_writer,用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)

    plotdata = {"batchsize": [], "loss": []}  # 存放批次值和损失值
    # 向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    # print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    print("cost:", cost.eval({X: train_X, Y: train_Y}))

    # 生成summary
    summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
    # 将summary写入文件
    summary_writer.add_summary(summary_str, epoch)


    # 训练完成后，保存模型
    saver.save(sess, "E://tensorflow_model//linermodel.cpkt")
    # 图形显示

    # plt.plot(train_X, train_Y, 'ro', label='Original data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    # plt.legend()
    # plt.show()
    #
    # plotdata["avgloss"] = moving_average(plotdata["loss"])
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss')
    # plt.title('Minibatch run vs. Training loss')
    #
    # plt.show()
    print("x = 0.2, z=", sess.run(z, feed_dict={X: 0.2}))

# 载入模型
# 代码不好用
# with tf.Session() as sess2:
    # saver.restore(sess2, "E://tensorflow_model//linear_regression.cpkt")
    # print("x = 0.2, z=", sess2.run(z, feed_dict={X: 0.2}))
