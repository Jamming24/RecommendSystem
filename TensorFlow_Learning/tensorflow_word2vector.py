# -*- coding:utf-8 -*-

import os
import numpy as np
from TensorFlow_Learning import cifar10_input
import tensorflow as tf
import pylab

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device("/cpu:0"):
    # vocab_size表示词表大小，每个词的向量个数为size个
    embedding = tf.get_variable("embedding", [vocab_size, size])
    # tf.nn.embedding_lookup方法只支持在CPU上运行， inputs是得到的词向量
    inputs = tf.nn.embedding_lookup(embedding, input_data)
