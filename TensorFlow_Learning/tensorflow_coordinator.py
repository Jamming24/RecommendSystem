# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import pylab
from TensorFlow_Learning import cifar10_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_coodinator():
    # 创建长度为100的队列
    queue = tf.FIFOQueue(100, "float")
    # 计数器
    c = tf.Variable(0.0)
    # 加1操作
    op = tf.assign_add(c, tf.constant(1.0))
    # 操作：将计数器的结果加入队列
    enqueue_op = queue.enqueue(c)
    # 创建一个队列管理器QueueRunner, 用这两个操作向q中添加元素，目前我们只使用一个线程
    qr = tf.train.QueueRunner(queue=queue, enqueue_ops=[op, enqueue_op])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        # 启动入队线程，Coordinator() 启动入队线程
        enqueue_threads = qr.create_threads(sess=sess, coord=coord, start=True)
        # 主线程
        for i in range(0, 10):
            print("-------------------------------")
            print(sess.run(queue.dequeue()))
        # 通知其他线程关闭，其他所有线程关闭之后，这一函数才能返回··
        coord.request_stop()


batch_size = 128
data_dir = "./cifar-10-binary/cifar-10-batches-bin"
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 定义协调器
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_batch, label_batch = sess.run([images_test, labels_test])
    print("_____\n", image_batch[0])
    print("_____\n", label_batch[0])
    pylab.imshow(image_batch[0])
    pylab.show()
    coord.request_stop()


