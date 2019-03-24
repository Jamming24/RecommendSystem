# -*- coding:utf-8 -*-

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 初始化学习参数
a = 0
# 初始化一切所提供的值
tf.constant_initializer(a)
# 从a到b均匀初始化
tf.random_uniform_initializer(a, b)
# 用所给平均值和标准差初始化均匀分布
tf.random_normal_initializer()
# 初始化常量
tf.constant_initializer(value=0, dtype=tf.float32)
# 正态分布随机数，均值mean，标准差stddev
tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
#########比较常用, 因为改函数具有截断功能，可以生成比较温和的初始值###########
# 截断正态分布随机数，均值mean，标准差stddev，不过只保留[mean-2*stddev, mean+2*stddev]范围内的随机数
tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
###############################################
# 均匀分布随机数，范围内[minval,maxval]
tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32)
# 满足均匀分布，但不影响输出数量级的随机值
tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32)
# 初始化为0
# 初始化为1
# 生成正交矩阵的随机数，当需要生成的参数是二维数，这个正交矩阵是由均匀分布的随机数矩阵经过SVD分解而来
tf.orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None)

# 一般的梯度下降算法的Optimizer
Optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent')
# 创建Adadelta优化器
Optimizer2 = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
# 创建Adagrad优化器
Optimizer3 = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1, use_locking=False,
                                       name='Adagrad')
# 创建momentum优化器，momentum：动量，一个tensor或者浮点值
moentum = tf.constant(3.0)
Optimizer4 = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=moentum, use_locking=False, name='Adagrad')
# 创建Adam优化器
Optimizer5 = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
                                    name='Adam')
# 创建FTRL优化器
Optimizer6 = tf.train.FtrlOptimizer(learning_rate=0.01, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                                    l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False,
                                    name='Ftrl')
# 创建RMSProp算法优化器
Optimizer7 = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
                                       name='RMSProp')


# 退化学习率方法
def exponential_decay(learning_rate, global_step, decay_steps, decay_rate):
    daceyed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    return daceyed_learning_rate


# 表示每1000步学习率缩减0.96
# learning_rate = exponential_decay(start_rate, global_step, 1000, 0.96)

# 退化学习率举例

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=10,
                                           decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
# 将global_step加一完成计步
add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    # 循环20步，将每步的学习率打印出来
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        print(g, rate)
