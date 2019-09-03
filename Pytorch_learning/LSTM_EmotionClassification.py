# -*- coding: utf-8 -*-
# @Time    : 2019/8/10 23:21
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : LSTM_EmotionClassification.py
# @Software: PyCharm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch


is_cuda = torch.cuda.is_available()


class EmbNet(nn.Module):
    def __init__(self, emb_size, hidden_size1, hidden_size2=400):
        super().__init__()
        self.embedding = nn.Embedding(emb_size, hidden_size1)
        self.fc = nn.Linear(hidden_size2, 3)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.fc(embeds)
        return F.log_softmax(out, dim=-1)


class IMDBRnn(nn.Module):
    def __init__(self, n_vocab, hidden_size, n_cat, bs=1, nl=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.nl = nl
        self.e = nn.Embedding(n_vocab, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, nl)
        self.fc2 = nn.Linear(hidden_size, n_cat)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.e(inp)
        h0 = c0 = Variable(e_out.data.new(*(self.nl, self.bs, self.hidden_size)).zero_())
        rnn_o, _ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.8)
        return self.softmax(fc)


def fit(epoch, model, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    running_loss = 0.0
    running_correct = 0
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.text, batch.label
        if is_cuda:
            text, target = text.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output, target)
        running_loss += F.nll_loss(output, target, size_average=False).item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = running_correct.__float__() / len(data_loader.dataset)
    print(
        f'第{epoch}次迭代, {phase} loss is {loss} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)},{accuracy}')
    return loss, accuracy


if __name__ == '__main__':
    TEXT = data.Field(lower=True, batch_first=False, fix_length=200)
    LABEL = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)
    # 构建词表
    TEXT.build_vocab(train, test, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
    LABEL.build_vocab(train)
    # print(LABEL.vocab.freqs)
    # print(TEXT.vocab.vectors)
    # print(TEXT.vocab.stoi)
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32)
    train_iter.repeat = False
    test_iter.repeat = False

    n_vocab = len(TEXT.vocab)
    n_hidden = 100
    # 为什么不收敛呢？？？？
    model = IMDBRnn(n_vocab=n_vocab, hidden_size=100, n_cat=3, bs=32)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_iter, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_iter, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)





