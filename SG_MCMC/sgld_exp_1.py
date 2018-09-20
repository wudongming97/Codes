import torch.nn as nn
from tensorboardX import SummaryWriter

from sgld import SGLD
from utils import *

INIT_LR = 0.01
LAST_LR = 0.000001
TOTAL_ITER_NUM = 500000


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

    def log_gaussian_piror(self):
        log_p = 0
        for param in self.parameters():
            log_p += log_gaussian(param, 0., 1.).sum()
        return log_p


criterion = nn.CrossEntropyLoss(reduction='sum')


def valid_acc(model, data_iter):
    acc = 0
    for x, y in data_iter:
        with torch.no_grad():
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_ = model(x)
            acc += get_cls_accuracy(y_, y)
    return acc / len(data_iter)


def train(model, trainer, train_iter, test_iter, log_dir):
    writer = SummaryWriter(log_dir='./runs/' + log_dir)
    n_batchs = len(train_iter)

    for t in range(TOTAL_ITER_NUM):
        x, y = next(iter(train_iter))
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        loss_px = criterion(model(x), y)
        loss_pw = -model.log_gaussian_piror() / n_batchs  # 增加高斯先验等价于L2_norm，注意要进行grad scale
        loss = loss_px + loss_pw
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        # update lr
        updated_lr = lr_sgld_scheduler(t, INIT_LR, LAST_LR, total_steps=TOTAL_ITER_NUM)
        for param_group in trainer.param_groups:
            param_group['lr'] = updated_lr

        if t % 100 == 0:
            # 查看权重分布
            acc = valid_acc(model, test_iter)
            writer.add_scalar('acc', acc.item(), )
            writer.add_scalar('loss', loss.item(), t)
            writer.add_scalar('lr', updated_lr, t)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), t)
            print('[%s] [T: %d] [Loss: %.3f] [Loss_px: %.3f] [Loss_pw: %.3f]' % (
                trainer.__class__.__name__, t, loss.item(), loss_px.item(), loss_pw.item()))


if __name__ == '__main__':
    # 用mlp分类模型上对比sgld和sgd的效果差异
    mnist_train_iter, mnist_test_iter = mnist_loaders('../../Datasets/MNIST/', 100)

    sgld_model = MLP()
    sgld_trainer = SGLD(sgld_model.parameters(), lr=INIT_LR)
    train(sgld_model, sgld_trainer, mnist_train_iter, mnist_test_iter, 'sgld')

    sgd_model = MLP()
    sgd_trainer = SGLD(sgd_model.parameters(), lr=INIT_LR, addnoise=False)
    train(sgd_model, sgd_trainer, mnist_train_iter, mnist_test_iter, 'sgd')
