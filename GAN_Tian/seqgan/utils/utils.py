# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        # 调用生成器进行生成
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w',encoding='utf-8') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer

    print('codes:' + codes)
    return codes


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    # 先使用MLE预训练一批
    supervised_g_losses = []
    data_loader.reset_pointer()
    # 遍历批次
    for it in range(data_loader.num_batch):
        # oracle数据中的下一批数据
        batch = data_loader.next_batch()
        # 进行模型的预训练
        _, g_loss = trainable_model.pretrain_step(sess=sess, x=batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
