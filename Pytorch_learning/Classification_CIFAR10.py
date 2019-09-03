# -*- coding: utf-8 -*-
# @Time    : 2019/8/11 10:50
# @Author  : Jamming
# @Email   : gaojiaming24@163.com
# @File    : Classification_CIFAR10.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def test_tensor():
    x = torch.empty(5, 3)
    y = torch.rand(5, 3)
    z = torch.zeros(5, 3, dtype=torch.long)
    print(x)
    print(y)
    print(z)
    print(x + y)
    print(torch.add(x, y))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 直接从GPU创建张量
        y = torch.ones_like(x, device=device)
        # 将张量移动到GPU中
        x = x.to(device)
        z = x + y
        print(z)
        print(y)


def test_grad():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    y = x + 2
    print(y)
    print(y.grad_fn)
    z = y * y * 3
    out = z.mean()
    print(out)
    print(z, out)
    print(out.backward())
    print(x.grad)
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# net = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# criterion = nn.MSELoss()
# optimizer.zero_grad()
# input = torch.randn(1, 1, 32, 32)
# output = net(input)
# target = torch.randn(10)
# target = target.view(1, -1)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
    print(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='.data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='.data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outsputs = net(inputs)
            loss = criterion(outsputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # plt.show()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))












