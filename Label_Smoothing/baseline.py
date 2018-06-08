import torch
import torch.nn as nn
import torch.optim as optim

from data import mnist_train_iter, mnist_test_iter
from model import init_weights
from utils import DEVICE, get_cls_accuracy, print_network

n_epochs = 10
lr = 2e-4
dropout_prob = 0.5

net = nn.Sequential(
    nn.Linear(784, 1024),
    nn.Dropout(dropout_prob, True),
    nn.ReLU(True),
    nn.Linear(1024, 1024),
    nn.Dropout(dropout_prob, True),
    nn.ReLU(True),
    nn.Linear(1024, 10)
).apply(init_weights).to(DEVICE)

print_network(net)

trainer = optim.Adam(net.parameters(), lr=lr, betas=[0.5, 0.99])
criterion = nn.CrossEntropyLoss()

# train
for e in range(n_epochs):
    net.train()
    for b, (x, l) in enumerate(mnist_train_iter):
        x = x.view(-1, 784).to(DEVICE)
        l = l.to(DEVICE)

        logits = net(x)
        loss = criterion(logits, l)

        net.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % 50 == 0:
            acc = get_cls_accuracy(logits, l)
            print('[%2d/%2d] loss: %.3f acc: %.3f' % (e + 1, n_epochs, loss.item(), acc.item()))

    # valid
    with torch.no_grad():
        net.eval()
        total_acc = 0
        for x, l in mnist_test_iter:
            x = x.view(-1, 784).to(DEVICE)
            l = l.to(DEVICE)
            logits = net(x)
            total_acc += get_cls_accuracy(logits, l).item()
        acc = total_acc / len(mnist_test_iter)
        print('在测试集上的准确率为：%.3f ' % acc)
