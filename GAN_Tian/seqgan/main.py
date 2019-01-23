# -*- coding:utf-8 -*-
import nltk
from GAN_Tian.seqgan.Seqgan import Seqgan

if __name__ =='__main__':
    # import nltk
    # nltk.download('punkt')
    gan = Seqgan()
    # train_model = 'real'  # oracle    cfg     real
    # local_data = 'data/en_demo2.txt'
    local_data = 'data/quora_50K_train.txt'
    gan.train_real(local_data)






