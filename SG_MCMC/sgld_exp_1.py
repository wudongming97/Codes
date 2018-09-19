import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from sgld import SGLD
from utils import *

mnist_train_iter, mnist_test_iter = mnist_loaders('../../Datasets/MNIST/', 100)


class MLP(nn.Module):
    """
    使用《Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks》中的网络结构
    """

    def __init__(self):
        super(MLP, self).__init__()
        self._block = nn.Sequential(
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 10))
        self.to(DEVICE)

    def forward(self, x):
        x = x.view(-1, 784)
        return self._block(x)


criterion = nn.CrossEntropyLoss()


def valid_acc(model):
    acc = 0
    for x, y in mnist_test_iter:
        with torch.no_grad():
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_ = model(x)
            acc += get_cls_accuracy(y_, y)
    return acc / len(mnist_test_iter)


def train(model, trainer, log_dir):
    lr_scheduler = optim.lr_scheduler.StepLR(trainer, step_size=10, gamma=0.65)
    writer = SummaryWriter(log_dir='./runs/' + log_dir)

    for epoch in range(200):
        lr_scheduler.step()
        for i, (x, y) in enumerate(mnist_train_iter):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss = criterion(model(x), y)
            trainer.zero_grad()
            loss.backward()
            trainer.step()

            if i % 100 == 0:
                print('[%s][Epoch: %d] [Batch: %d] [Loss: %.3f]' % (
                    trainer.__class__.__name__, epoch, i, loss.item()))

        # 查看权重分布
        acc = valid_acc(model)
        writer.add_scalar('acc', acc.item(), epoch)
        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('lr', lr_scheduler.get_lr()[0], epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


if __name__ == '__main__':
    # 用mlp分类模型上对比sgld和sgd的效果差异
    
    sgld_model = MLP()
    sgld_trainer = SGLD(sgld_model.parameters(), lr=0.01)
    train(sgld_model, sgld_trainer, 'sgld')

    sgd_model = MLP()
    sgd_trainer = SGLD(sgd_model.parameters(), lr=0.01, addnoise=False)
    train(sgd_model, sgd_trainer, 'sgd')
