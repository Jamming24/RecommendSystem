# -*- coding:utf-8 -*-

import tensorflow as tf

var1 = tf.Variable(1.0, name='firstwar')
print("var1:", var1.name)
var1 = tf.Variable(2.0, name='firstwar')
print("var1:", var1.name)
var2 = tf.Variable(3.0)
print("var2:", var2.name)
var2 = tf.Variable(4.0)
print("var1:", var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=", var1.eval())
    print("var2=", var2.eval())
#
# 使用get_variable()
get_val1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
print("get_val1:", get_val1.name)

get_val1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(0.4))
print("get_var1:", get_val1.name)

with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    print("get_var1:", get_val1.eval())

# 使用get_variable 和variable_scope
with tf.variable_scope("test1",): # 定义一个作用域test1
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2",): # 定义一个作用域test2
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
print("var1:", var1.name)
print("var2:", var2.name)

# 共享变量功能实现
with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
print("var3:", var3.name)
print("var4:", var4.name)

# with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4)):
#     var1 = tf.get_variable_scope("firstwar", shape=[2], dtype=tf.float32)
#     with tf.get_variable_scope("test2"):
#         var2 = tf.get_variable_scope("firstvar", shape=[2], dtype=tf.float32)
#         var3 = tf.get_variable_scope("var3", shape=[2], initializer=tf.constant_initializer(0.3))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=", var1.eval()) # 作用域test1下的变量
    print("var2=", var2.eval()) # 作用域test2下的变量 继承test1的初始化
    print("var3=", var3.eval()) # 作用test2下的变量

