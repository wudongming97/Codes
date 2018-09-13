import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from tensorboardX import SummaryWriter

from utils import *


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self._block = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10))
        self.to(DEVICE)

    def forward(self, x):
        x = x.view(-1, 784)
        return self._block(x)


mnist_train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=100,
    shuffle=True,
)

mnist_test_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=1000,
    drop_last=True,
)

criterion = nn.CrossEntropyLoss()


def test(model):
    acc = 0
    model.eval()
    for x, y in mnist_test_iter:
        with torch.no_grad():
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_ = model(x)
            acc += get_cls_accuracy(y_, y)
    return acc / len(mnist_test_iter)


def train(trainer_cls):
    model = MLP()
    trainer = trainer_cls(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(trainer, gamma=0.998)
    writer = SummaryWriter(log_dir='./run/' + trainer_cls.__name__)

    for epoch in range(100):
        n_batchs = len(mnist_train_iter)
        for i, (x, y) in enumerate(mnist_train_iter):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss = criterion(model(x), y)
            trainer.zero_grad()
            loss.backward()
            trainer.step()

            if i % 20 == 0:
                acc = test(model)
                lr_scheduler.step()
                cur_step = epoch * n_batchs + i

                writer.add_scalar('acc', acc.item(), cur_step)
                writer.add_scalar('loss', loss.item(), cur_step)
                writer.add_scalar('lr', lr_scheduler.get_lr()[0], cur_step)

            if i % 100 == 0:
                print('[%s][Epoch: %d] [Batch: %d] [Loss: %.3f]' % (trainer_cls.__name__, epoch, i, loss.item()))
