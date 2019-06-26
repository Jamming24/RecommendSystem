# -*- coding:utf-8 -*-

import os
import numpy as np
from TensorFlow_Learning import cifar10_input
import tensorflow as tf
import pylab

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 使用卷积神经网络做图片分类
def test_load_data():
    # 取数据
    batch_size = 128
    # 绝对路径
    # data_dir = 'D:/PycharmWorkSpace/TensorFlow_Learning/cifar-10-binary/cifar-10-batches-bin'
    # 相对路径
    data_dir = "./cifar-10-binary/cifar-10-batches-bin"
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # 队列线程启动代码， 如果没有这行，将处于一个挂起状态
    tf.train.start_queue_runners()
    image_batch, label_batch = sess.run([images_test, labels_test])
    print("__\n", image_batch[0])
    print("__\n", label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()


def test_show_picture():
    filename = "./cifar-10-binary/cifar-10-batches-bin/test_batch.bin"
    bytestream = open(filename, "rb")
    buf = bytestream.read(10000 * (1+32*32*3))
    bytestream.close()

    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(10000, 1+32*32*3)
    labels_images = np.hsplit(data, [1])
    labels = labels_images[0].reshape(10000)
    images = labels_images[1].reshape(10000, 32, 32, 3)

    # 导出第一幅图
    img = np.reshape(images[0], (3, 32, 32))
    img = img.transpose(1, 2, 0)

    print(labels[0])
    pylab.imshow(img)
    pylab.show()

    # 对图片进行其他处理
    height = 32
    width = 32
    reshaped_image = 1000
    # tf.random_crop(),为图片随机裁剪
    distored_image = tf.random_crop(reshaped_image, [height, width, 3])
    # 随机左右翻转
    distored_image = tf.image.random_flip_left_right(distored_image)
    # 随机对亮度变化
    distored_image = tf.image.random_brightness(distored_image, max_delta=63)
    # 随机对比度变化
    distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)
    # 减去均值像素， 并除以像素方差（图片标准化）
    float_image = tf.image.per_image_standardization(distored_image)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 返回map_feature 卷积之后提取到的特征
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 2x2的池化窗口, 需要进行padding操作
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')


def CNN_with_Global_Average_Pooling_Layer():
    # 本例子使用了全局平均池化层来代替传统的全连接层，使用3个卷积层的同卷积操作，滤波器为5x5，
    # 每个卷积层后面都会跟两个步长为2x2的池化层，滤波器为2x2，2层的卷积加上池化后是输出10个通道的卷积
    batch_size = 128
    data_dir = "./cifar-10-binary/cifar-10-batches-bin"
    print("begin")
    images_train, labels_train = cifar10_input.inputs(eval_data=False, data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
    # 定义网络结构 ##################################
    # 定义占位符 cifar data为24*24*3 大小 其中3表示三通道
    x = tf.placeholder(tf.float32, [None, 24, 24, 3])
    # 0~9的数据分类
    y = tf.placeholder(tf.float32, [None, 10])
    # 由于使用64个三通道的5x5大小的卷积核 所以参数tensor大小如下
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    x_image = tf.reshape(x, [-1, 24, 24, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 64的定义是什么？？？？
    W_conv3 = weight_variable([5, 5, 64, 10])
    b_conv3 = bias_variable([10])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    nt_hpool3 = avg_pool_6x6(h_conv3)
    nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
    y_conv = tf.nn.softmax(nt_hpool3_flat)

    cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
    # 使用AdamOptimizer 学习率为0.0001
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 这行具体是干什么的？？？？？
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 队列线程启动代码， 如果没有这行，将处于一个挂起状态
    tf.train.start_queue_runners(sess=sess)
    for i in range(10000):
        image_batch, label_batch = sess.run([images_train, labels_train])
        # one_hot编码
        label_b = np.eye(10, dtype=float)[label_batch]

        train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)

        if i % 200 == 0:
            train_accuray = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
            print("step %d, training accuracy %g" % (i, train_accuray))


def test_conv2d_transpose():
    # 反卷积举例说明
    # 反卷积不能复原卷积操作的输入值， 仅仅是将卷积变换过程中的步骤反向变换一次而已
    # 通过将卷积核转置，与卷积后的结果再做一遍卷积，所有反卷积又叫转置卷积
    # 反卷积具体操作如下：
    # （1）首先是将卷积核反转，不是线性代数的转值操作，而是，上下左右方向进行递序操作
    # （2）再将卷积结果作为输入，进行补零（padding）操作，即往每一个元素的后面补零，每一个元素沿着步长方向补零
    # （3）在扩充后的输入基础上在对整体补零，以原始输入的shape作为输出， 按照前面介绍的介绍的卷积padding原则，
    #  计算padding的补零位置以及个数，得到的补零位置要上下和左右各自颠倒一下
    # （4）将补零后的卷积结果作为真正的输入，反卷积后的卷积核为filter，进行步长为1的卷积操作
    value = 0 # 代表卷积之后操作的张量， 一般用NHWC类型
    filters = 0 # 代表卷积核
    output_shape = 0 # 代表输出的张量形状也是一个四维张量
    strides = 0 # 代表步长
    # padding 代表原数据生成value时使用的补零方式，是用来检查输入形状与输出形状是否合规的
    # return 反卷积后的结果， 按照output_shape指定的形状
    # tf.nn.conv2d_transpose(value=value, filter=filters, output_shape=output_shape, strides=strides,
    #                        padding="SAME", data_format="NHWC", name="反卷积")
    # NHWC类型是神经网络中处理图像方面常用的类型， 4个字母代表，N:个数，H:高，W:宽，C:通道数

    # 模拟数据
    img = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1]))
    filter = tf.Variable(tf.constant([1.0, 0, -1, -2], shape=[2, 2, 1, 1]))
    # 分别进行VALID与SAME操作
    conv = tf.nn.conv2d(img, filter=filter, strides=[1, 2, 2, 1], padding='VALID')
    cons = tf.nn.conv2d(img, filter=filter, strides=[1, 2, 2, 1], padding='SAME')
    print(conv.shape)
    print(cons.shape)
    # 再进行反卷积output_shape=[]跟输入的尺寸相同
    contv = tf.nn.conv2d_transpose(value=conv, filter=filter, output_shape=[1, 4, 4, 1], strides=[1, 2, 2, 1],
                                   padding='VALID')
    conts = tf.nn.conv2d_transpose(value=cons, filter=filter, output_shape=[1, 4, 4, 1], strides=[1, 2, 2, 1],
                                   padding="SAME")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("conv:\n", sess.run([conv, filter]))
        print("cons:\n", sess.run([cons, filter]))
        print("contv:\n", sess.run([contv, filter]))
        print("conts:\n", sess.run([conts, filter]))
        # [-2, 0, -2, 0]
        # [2, 4, 2, 4]
        # [-2, 0, -2, 0]
        # [2, 4, 2, 4]
        # 输出一列对应上述一行


# def test_Anti_pooling():
    # 反池化举例

if __name__ == '__main__':
    # 本例子使用了全局平均池化层来代替传统的全连接层，使用3个卷积层的同卷积操作，滤波器为5x5，
    # 每个卷积层后面都会跟两个步长为2x2的池化层，滤波器为2x2，2层的卷积加上池化后是输出10个通道的卷积
    # CNN_with_Global_Average_Pooling_Layer()
    # 反卷积举例
    test_conv2d_transpose()



