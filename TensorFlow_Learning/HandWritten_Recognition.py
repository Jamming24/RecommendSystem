# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 下载手写数字识别数据集 到MNIST_data目录下
# 将样本转化为one_hot编码
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print('输入数据:', mnist.train.images)
# print('输入训练数据打印shape:', mnist.train.images.shape)
# print('输入测试数据打印shape:', mnist.test.images.shape)
# print('交叉验证集打印shape:', mnist.validation.images.shape)
#
import pylab
# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()

tf.reset_default_graph()
# 定义占位符 MMIST数据的维度是28x28=784
# x表示能够输入任意数量的MNIST图像，每张图都是784维的向量，None表示第一维度可以是任意长度的
x = tf.placeholder(tf.float32, [None, 784])
# 数字0~9，共10个类别
y = tf.placeholder(tf.float32, [None, 10])
# 模型需要权重值和偏置量，他们被统一叫做学习参数，在tensorflow里使用Variable定义学习参数
# 一个Variable代表一个可以修改的张量，在tensorflow的图中  本身也是一个变量
# W 设置为随机值， b设置为零
W = tf.Variable(tf.random_normal([784, 10]))  # 为啥是第二个参数是10呢？？？？？？
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax分类
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 25
batch_size = 100
display_step = 1
# 模型保存参数
saver = tf.train.Saver()
model_path = "log/HandWritten_recognition.ckpt"
# # 启动session
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     # 启动循环开始训练
#     for epoch in range(training_epochs):
#         avg_cost = 0
#         total_batch = int(mnist.train.num_examples / batch_size)
#         # 使用随机批梯度下降循环所以数据集
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             # 运行优化器
#             c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
#             # 计算平均loss值
#             # print(type(c))
#             # print(c)
#             avg_cost += c[1] / total_batch
#         # 显示训练中的详细信息
#         if (epoch + 1) % display_step == 0:
#             # '%04d' % (epoch+1) 表示格式化输出 4表示 总计位数
#             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
#     print("Training Finished!")
#     # 测试Model  用tf.arg_max返回onehot编码中数值为1那个元素的下标 这三行代码都是干什么用的呢？？？？？？？
#     correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
#     # 计算准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
#     # 保存模型,并将模型保存的路径打印出来
#     save_path = saver.save(sess, model_path)
#     print("Model saved in file: %s" % save_path)


# 读取和测试模型
print("Starting 2nd session.....")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)
    # 测试model
    correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

    output = tf.arg_max(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
    print(outputval, predv, batch_ys)
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

