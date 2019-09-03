# -*- coding: utf-8 -*-
# @Time    : 2019/8/11 23:22
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Pytorch_tensor.py
# @Software: PyCharm
import torch
from torch.autograd.function import Function
import numpy as np


def test_torch():
    print(torch.__version__)
    x = torch.rand(2, 3)
    print(x)
    print(x.shape)
    print(x.size())
    y = torch.rand(2, 3, 4, 5)
    print(y.size())
    print(y)

    scalar = torch.tensor(3.1433223)
    print(scalar.item())
    x = torch.randn(3, 3)
    # 沿着行取最大值
    max_value, max_idx = torch.max(x, dim=1)
    print(x)
    print(max_value)
    print(max_idx)
    y = torch.randn(3, 3)
    x.add_(y)
    print(x)


def test_autograd():
    x = torch.rand(5, 5, requires_grad=True)
    print(x)
    y = torch.rand(5, 5, requires_grad=True)
    print(y)
    z = torch.sum(x+y)
    print(z)
    print(x.grad)
    print(y.grad)
    print(x.grad, y.grad)
    z.backward()
    print(x.grad, y.grad)
    x = torch.rand(5, 5, requires_grad=True)
    y = torch.rand(5, 5, requires_grad=True)
    z = x**2 + y**3
    print(x.grad)
    z.backward(torch.ones_like(x))
    print(x.grad)
    # 使用with torch.no_grad()上下文管理器临时禁止对已设置requires_grad=True的张量进行自动求导。这个方法在测试集计算准确率的时候会经常用到
    with torch.no_grad():
        print((x+y*2).requires_grad)


class MulConstant(Function):
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return tensor * constant

    def backward(ctx, grad_outputs):
        return grad_outputs, None



