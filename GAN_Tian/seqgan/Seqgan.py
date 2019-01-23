# -*- coding:utf-8 -*-
import json
from time import time

from seqgan.Gan import Gan
from seqgan.SeqganDataLoader import DataLoader, DisDataloader
from seqgan.SeqganDiscriminator import Discriminator
from seqgan.SeqganGenerator import Generator
from seqgan.SeqganReward import Reward
# 评价和文本处理的代码
from seqgan.utils.metrics.Cfg import Cfg
from seqgan.utils.metrics.EmbSim import EmbSim
from seqgan.utils.metrics.Nll import Nll
from seqgan.utils.oracle.OracleCfg import OracleCfg
from seqgan.utils.oracle.OracleLstm import OracleLstm
from seqgan.utils.text_process import *
from seqgan.utils.utils import *


class Seqgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 2000  # 词库大小，被设定为5000
        self.emb_dim = 32  # 32
        self.hidden_dim = 32  # 32
        self.sequence_length = 20  # 序列长度
        self.filter_size = [2, 3]  # 两层核大小
        self.num_filters = [100, 200]  # 两层核个数，即通道数量
        self.l2_reg_lambda = 0.2  # 正则化参数
        self.dropout_keep_prob = 0.75  # 用于随机丢弃的节点比例
        self.batch_size = 32  # 批大小
        self.generate_num = 1024 # 生成样例的数量，被设定为10000
        self.start_token = 0  # 开始的时候的选取的令牌值
        self.rollout_num = 16

        self.pre_epoch_num = 10  # 预训练的轮数 生成和判别共用      #  80
        self.adversarial_epoch_num = 50  # 对抗训练的轮数    # 100
        self.adversarial_epoch_dis_num = 15  # 一次對抗中，進行幾次判別  # 15原始


        # self.positive = 'data/msrp_demo_positive.txt'
        # self.positive = 'data/quora_50K_positive.txt'
        self.positive = 'data/quora_1024_positive.txt'
        self.positive_file = 'save/positive.txt'  # 原始句对用于判别的正例文件
        self.negitive_file = 'save/negitive.txt'  # 原始句子+生成的句子构成的反例文件
        self.oracle_file = 'save/oracle.txt'  # 原始的文件
        self.generator_file = 'save/generator.txt'  # 生成的令牌文件
        self.test_file = 'save/test_file.txt'  # 生成的用来测试的文件

    # 初始化评价指标 Nll inll DocEmbSim
    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from seqgan.utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
                           num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    #  合并出负例——改
    def combin_orc_gen(self):
        with open(self.negitive_file, 'w', encoding='utf-8') as negw:
            with open(self.oracle_file, 'r', encoding='utf-8') as orar:
                with open(self.generator_file, 'r', encoding='utf-8') as genr:
                    for x, y in zip(list(orar), list(genr)):
                        negw.write(x.strip() + ' ' + y.strip() + '\n')

    # 用于reward的样例生成组合
    def combin_orc_gen_reward(self, gen):
        result = []
        with open(self.oracle_file, 'r', encoding='utf-8') as orar:
            for x, y in zip(list(orar), list(gen)):
                result.append(x.strip() + ' ' + y.strip())
        return result

    # 训练判别器
    def train_discriminator(self):
        # 将生成器生成的文件写入generator_file
        generate_samples(sess=self.sess, trainable_model=self.generator, batch_size=self.batch_size,
                         generated_num=self.generate_num, output_file=self.generator_file)

        ## 加载数据进行判别，oracle_file为正例，generator_file为反例
        # self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        #  加载数据进行判别，——改
        self.combin_orc_gen()
        self.dis_data_loader.load_train_data(self.positive_file, self.negitive_file)

        # 貌似分成了三批进行计算判别，写死了的@@@@@@@@@@@@@@@@@这个数据可以修改
        # print('1-16批判别损失：')
        for i in range(16):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            # 喂数据
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            # 计算损失
            loss, _ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)
            # print(loss, end=' ')

    # 评价程序
    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs, ')
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    # 初始化真实数据的训练配置
    def init_real_trainng(self, data_loc=None):
        from seqgan.utils.text_process import text_precess, text_to_code, text_to_code_p
        from seqgan.utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        # 先给他禁了，这个重置了词库大小和序列长度，不能禁止这个，——改
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        print('sequence_length:',self.sequence_length)
        print('vocab_size:',self.vocab_size)

        # 设定生成器和判别器的各项参数
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)
        #  判别器的序列长度需要乘以二，生成器不需要改——改
        discriminator = Discriminator(sequence_length=self.sequence_length * 2, num_classes=2,
                                      vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        # 设置加载数据的各种加载器参数
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None

        #  判别器的序列长度需要乘以二，生成器不需要改——改
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length * 2)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        # 把data_loc文件中的句子变成单词 list
        tokens = get_tokenlized(data_loc)
        # 单词去重，变set
        word_set = get_word_list(tokens)
        # 使用原始的数据文档构建了两个字典，没有固定大小，单词-索引，和索引-单词
        [word_index_dict, index_word_dict] = get_dict(word_set)
        # 将原始文件写出为oracle_file
        with open(self.oracle_file, 'w', encoding='utf-8') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))

        # 将正例文件token化，写出来正例文件，用于判别器——改
        token_positive = get_tokenlized(self.positive)
        with open(self.positive_file, 'w', encoding='utf-8') as pfile:
            pfile.write(text_to_code_p(token_positive, word_index_dict, self.sequence_length))

        return word_index_dict, index_word_dict

    # 初始化真实数据的评价指标
    def init_real_metric(self):
        from seqgan.utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
                           num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    # 开始真实数据的训练
    def train_real(self, data_loc=None):
        from seqgan.utils.text_process import code_to_text
        from seqgan.utils.text_process import get_tokenlized
        # 调用配置函数，配置并初始化判别器和生成器，并得到原始数据获得的 单词-索引 和 索引-单词 字典
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        # 初始化真实数据的评价指标
        self.init_real_metric()

        # 读取生成的代码文件generation_file，通过字典将其翻译为文本写入test_file
        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r', encoding='utf-8') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w', encoding='utf-8') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        # 初始化全局变量
        self.sess.run(tf.global_variables_initializer())

        # self.pre_epoch_num = 80
        # self.adversarial_epoch_num = 100

        # 打开日志文件写入
        self.log = open('experiment-log-seqgan-real.csv', 'w')
        # 将生成器的生成的文件写入generator_file
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        # 建立真实数据的批次
        self.gen_data_loader.create_batches(self.oracle_file)
        # 开始预训练生成文本
        print('======================================================================================================')
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            # print('generator epoch:' + str(self.epoch) + '\t time:' + str(end - start)+'\t loss:'+loss)
            print('generator epoch:' + str(epoch) + '\t time:' + str(end - start) + '\t loss:' + str(loss))
            self.add_epoch()
            if epoch % 5 == 0:
                # 每训练5批就将生成器的生成的文件写入generator_file
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                # 将code转换成文本
                get_real_test_file()
                self.evaluate()

        # 开始预训练判别
        print('======================================================================================================')
        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            # print('\ndiscriminator epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('\n======================================================================================================')
        print('adversarial training:')
        # 奖励的配置
        self.reward = Reward(self.generator, update_rate=0.8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            # 这个index有个毛的用处
            for index in range(1):
                # 又从0开始生成了一遍
                samples = self.generator.generate(self.sess)

                # 将生成的负样例与原始句子进行组合作为训练预料——改——先不动--這個問題在get_reward()函數內部解決了
                # sample_negative = self.combin_orc_gen_reward(gen=samples)
                # rollout_num 延迟收益中的过程序列数，即进行16步之后再计算收益，过程存储在中间变量里
                rewards = self.reward.get_reward(self.sess, samples, rollout_num=self.rollout_num, discriminator=self.discriminator)
                # 先不动
                # rewards = self.reward.get_reward(self.sess, sample_negative, rollout_num=16, discriminator=self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print('\n第', index, '次reward更新损失：', loss)
                # print(loss)
            end = time()
            self.add_epoch()
            # print('adversarial epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            print('adversarial epoch:' + str(epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, trainable_model=self.generator, batch_size=self.batch_size,
                                 generated_num=self.generate_num, output_file=self.generator_file)
                get_real_test_file()  # 打开生成的token文件，转换成文本文件
                self.evaluate()

            self.reward.update_params()
            # 貌似每进行一次对抗就进行了15次判别
            for _ in range(self.adversarial_epoch_dis_num):
                # print('\ntrain_discriminator_', _ + 1)
                self.train_discriminator()
