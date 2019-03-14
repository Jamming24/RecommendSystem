# -*- coding:utf-8 -*-

import os

import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plotdata = {"batchsize": [], "loss": []}

c = tf.constant(0.0)
# tf.Graph().as_default() 表示用tf.Graph()建立一个图
g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    # 通过名字获取对应的元素
    print(c1.name) # 获取名字
    t = g.get_tensor_by_name(name="Const:0")
    print(t)
    ####################
    print(g)
    print(c.graph)
    g2 = tf.get_default_graph()
    print(g2)

    # tf.reset_default_graph()
    g3 = tf.get_default_graph()
    print(g3)

g3 = tf.get_default_graph()

a = tf.constant([[1., 2.]])
b = tf.constant([[1.], [3.]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(">>>>>>>")
print(tensor1.name, tensor1)
print("<<<<<<<")

test = g3.get_operation_by_name("exampleop")
print(">>>>>>>>>>>>>>>>>>>>")
print(test)
print("<<<<<<<<<<<<<<<<<<<<")

print(tensor1.op.name)
print(">>>>>>>>>>>>>>>>>>>>")
testop = g3.get_operation_by_name("exampleop")
print(testop)
print("<<<<<<<<<<<<<<<<<<<<<")

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    # tf.get_default_graph() 获取当前图
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)

# 获取图中的全部元素
tt2 = g.get_operations()
print(tt2)
# 函数as_graph_element获取了t1的真实张量对象，并且赋值给了tt3
tt3 = g.as_graph_element(c1)
print(tt3)
