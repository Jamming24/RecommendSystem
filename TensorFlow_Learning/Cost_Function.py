# -*- coding:utf-8 -*-

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 标签值
labels = [[0, 0, 1], [0, 1, 0]]
# labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
# 网络输出值
logits = [[2, 0.5, 6], [0.1, 0, 3]]
logits_scaled = tf.nn.softmax(logits=logits)
logits_scales2 = tf.nn.softmax(logits=logits_scaled)

# 计算logits和targets的交叉熵 Logits和labels必须为相同的shape的数据类型
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    print("Scaled=", sess.run(logits_scaled))
    print("Scaled2=", sess.run(logits_scales2))
    # 正确的方式
    print("rell=", sess.run(result1))
    print("rel2=", sess.run(result2))
    print("rel3=", sess.run(result3))
    print("rel4=", sess.run(result4))

# sparse交叉熵的使用
# 表明labels中总共分为3个类：0、1、2
labels = [2, 1]
loss = tf.reduce_mean(result1)
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5=", sess.run(result5))
    print("loss=", sess.run(loss))
