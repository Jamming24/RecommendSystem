# -*- coding:utf-8 -*-

import os
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
    tf.train.start_queue_runners()
    image_batch, label_batch = sess.run([images_test, labels_test])
    print("__\n", image_batch[0])
    print("__\n", label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()


filename = "./cifar-10-binary/cifar-10-batches-bin/test_batch.bin"
bytestream = open(filename, "rb")
buf = bytestream.read(10000 * (1+32*32*3))
bytestream.close()
