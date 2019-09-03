# -*- coding: utf-8 -*-
# @Time    : 2019/8/20 10:14
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : LanguageModeling.py
# @Software: PyCharm

import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data as d
from torchtext import datasets
from torchtext.vocab import GloVe
# import model


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    lstm.eval()
    total_loss = 0
    hidden = lstm.init_hidden(batch_size)
    for batch in data_source:
        data, targets = batch.text,batch.target.view(-1)
        output, hidden = lstm(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0]/(len(data_source.dataset[0].text)//batch_size)


def trainf():
    # Turn on training mode which enables dropout.
    lstm.train()
    total_loss = 0
    start_time = time.time()
    hidden = lstm.init_hidden(batch_size)
    for i,batch in enumerate(train_iter):
        data, targets = batch.text, batch.target.view(-1)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        lstm.zero_grad()
        output, hidden = lstm(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(lstm.parameters(), clip)
        for p in lstm.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            (print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, i, len(train_iter), lr,elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss))))
            total_loss = 0
            start_time = time.time()


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super().__init__()
        self.drop = nn.Dropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        s = output.size()
        decoded = self.decoder(output.view(s[0] * s[1], s[2]))
        return decoded.view(s[0], s[1], decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


os.environ['TORCH_HOME'] = 'E:/pytorch_data/'
is_cuda = torch.cuda.is_available()
TEXT = d.Field(lower=True, batch_first=True,)
train, valid, test = datasets.WikiText2.splits(TEXT,root='E:/pytorch_data/data')
batch_size=20
bptt_len=30
clip = 0.25
lr = 20
log_interval = 200
train[0].text = train[0].text[:(len(train[0].text)//batch_size)*batch_size]
valid[0].text = valid[0].text[:(len(valid[0].text)//batch_size)*batch_size]
test[0].text = test[0].text[:(len(valid[0].text)//batch_size)*batch_size]
# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0])['text'][0:10])
TEXT.build_vocab(train)
train_iter, valid_iter, test_iter = d.BPTTIterator.splits((train, valid, test), batch_size=batch_size, bptt_len=bptt_len, device=0,repeat=False)
criterion = nn.CrossEntropyLoss()
emsize = 200
nhid=200
nlayers=2
dropout = 0.2

ntokens = len(TEXT.vocab)
lstm = RNNModel(ntokens, emsize, nhid,nlayers, dropout, 'store_true')
if is_cuda:
    lstm = lstm.cuda()

best_val_loss = None
epochs = 40

for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    trainf()
    val_loss = evaluate(valid_iter)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                   val_loss, math.exp(val_loss)))
    print('-' * 89)
    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0