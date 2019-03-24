# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_Convolution_kernel():
    # 定义张量
    # [batch, in_height, in_width, in_channels] [训练时一个批次的图片数量, 图片高度，图片宽度，图像通道数]
    # 1.0表示初始值都为1.0
    input = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 1]))
    input2 = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 2]))
    input3 = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1]))
    # 定义卷积核
    # [filter_height, filter_width, in_channels, out_channels]
    # 卷积核的高度， 卷积核的宽度， 图像通道数（决定输入通道数目），卷积核个数（决定输出feature map的数目）
    filter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))
    filter2 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 1, 2]))
    filter3 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 1, 3]))
    filter4 = tf.Variable(
        tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 2, 2]))
    filter5 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 2, 1]))
    # 定义卷积操作， padding的值为‘VALID’，表示边缘不填充， 当其为‘SAME’时，表示填充到卷积核可以到达图像边缘
    # strides表示卷积时在图像每一维的步长,这是一个一维的向量，长度为4
    op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME')
    op2 = tf.nn.conv2d(input, filter2, strides=[1, 2, 2, 1], padding='SAME')
    op3 = tf.nn.conv2d(input, filter3, strides=[1, 2, 2, 1], padding='SAME')
    op4 = tf.nn.conv2d(input2, filter4, strides=[1, 2, 2, 1], padding='SAME')
    op5 = tf.nn.conv2d(input2, filter5, strides=[1, 2, 2, 1], padding='SAME')
    op6 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME')
    vop6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')
    vop1 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # 卷积核输出 按列排列 窄卷积和同卷积（步长唯一并且补零操作的卷积）
        print('op1:', sess.run([op1, filter1]))  # 1-1 一个输入一个输出
        print("-------------------------")
        print('op2:', sess.run([op2, filter2]))  # 1-2 多卷积核 按列取
        print('op3:', sess.run([op3, filter3]))  # 1-3 一个输入，三个输出
        print("---------------------------")
        print('op4:', sess.run([op2, filter2]))
        print('op5:', sess.run([op2, filter2]))
        print("----------------------------")
        print("op1:", sess.run([op1, filter1]))  # 1-1 一个输入 一个输出
        print("vop1:", sess.run([vop1, filter1]))
        print("op6:", sess.run([op6, filter1]))
        print("Vop6:", sess.run([vop6, filter1]))


def extracting_image_contour():
    # 使用sobel卷积操作 提取图片轮廓
    # 读取图片
    myimg = mpimg.imread('img.jpg')
    # 显示图片
    # plt.imshow(myimg)
    # # 不显示坐标轴
    # plt.axis('off')
    # plt.show()
    # (640, 640, 3)
    print(myimg.shape)
    full = np.reshape(myimg, [1, 640, 640, 3])
    inputfull = tf.Variable(tf.constant(1.0, shape=[1, 640, 640, 3]))
    # 定义卷积核
    filter = tf.Variable(tf.Variable(tf.constant([[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                                  [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                                  [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
                                                 shape=[3, 3, 3, 1])))
    # 卷积操作，三个通道输入， 生成一个feature map 同卷积操作
    op = tf.nn.conv2d(inputfull, filter, strides=[1, 1, 1, 1], padding='SAME')
    o = tf.cast((op - tf.reduce_min(op) / (tf.reduce_max(op) - tf.reduce_min(op))) * 255, tf.uint8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t, f = sess.run([o, filter], feed_dict={inputfull: full})
        t = np.reshape(t, [640, 640])
        # 显示图片
        plt.imshow(t, cmap='Greys_r')
        # plt.axis('off')
        plt.show()


def use_Pooling_funtion():
    # tensorflow里的池化函数 包括最大池化和均值池化
    # tf.nn.max_pool(value=, ksize=, strides=, padding=, name=None)
    # tf.nn.avg_pool(value=, ksize=, strides=, padding=, name=None)
    # value:需要池化的输入， 一般池化层接在卷积层后面，所以通常输入的是feature map，依然是[batch, height, width, channels]的shape
    # ksize:池化窗口的大小， 取一个四维向量， 一般是[1, height, width, 1], 因为我们不想在batch和channels上做池化，所以这两个维度设置为1
    # strides:窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    # padding:和卷积的参数一样，也是取VALID和SAME， VALID不是paddding操作， SAME是padding操作

    # 手动生成一个4x4的矩阵来模拟图片
    img = tf.constant([[[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
                       [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],
                       [[2.0, 6.0], [2.0, 6.0], [2.0, 6.0], [2.0, 6.0]],
                       [[3.0, 7.0], [3.0, 7.0], [3.0, 7.0], [3.0, 7.0]]])
    img = tf.reshape(img, [1, 4, 4, 2])
    # 定义池化操作---------------------------------
    # 最大池化操作
    pooling = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    pooling1 = tf.nn.max_pool(value=img, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    # 均值池化操作
    pooling2 = tf.nn.avg_pool(value=img, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')
    pooling3 = tf.nn.avg_pool(value=img, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # 取均值操作
    nt_hpool2_flat = tf.reshape(tf.transpose(img), [-1, 16])
    # 1 表示对行求平均值， 0表示对列求平均值
    pooling4 = tf.reduce_mean(nt_hpool2_flat, 1)
    # -----------------------------------------
    # 运行池化操作
    with tf.Session() as sess:
        print("image:")
        image = sess.run(img)
        print(image)
        print("result:", sess.run(pooling))
        print("result1:", sess.run(pooling1))
        print("result2:", sess.run(pooling2))
        # pooling3是常用的操作手法 叫做全局池化法，结果与result4均值操作一样
        print("result3:", sess.run(pooling3))
        flat, result4 = sess.run([nt_hpool2_flat, pooling4])
        print("result4:", result4)
        print("flat:", flat)


if __name__ == '__main__':
    # 卷积核举例
    # test_Convolution_kernel()
    # 使用sobel卷积操作 提取图片轮廓
    # extracting_image_contour()
    # 池化函数的使用举例
    use_Pooling_funtion()


