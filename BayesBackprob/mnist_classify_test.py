import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from tensorboardX import SummaryWriter

from bblayer import BBLayer
from utils import *

mnist_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=100,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

test_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=1000,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


def test(model, data_iter):
    # test
    acc = 0
    bb_mlp.eval()
    for x, y in data_iter:
        with torch.no_grad():
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            y_ = model(x, True)
            acc += get_cls_accuracy(y_, y)
    return acc / len(data_iter)
    # print('[Test] acc: %.3f' % (acc / len(test_iter)))


class BBMlp(nn.Module):
    def __init__(self):
        super(BBMlp, self).__init__()
        self.fc1 = BBLayer(784, 1200)
        self.fc2 = BBLayer(1200, 1200)
        self.fc3 = BBLayer(1200, 10)
        self.to(DEVICE)

    def forward(self, x, infer=False):
        x = self.fc1(x, infer)
        x = torch.relu(x)
        x = self.fc2(x, infer)
        x = torch.relu(x)
        x = self.fc3(x, infer)
        return x

    def kl_loss(self):
        return self.fc1.kl_loss + self.fc2.kl_loss + self.fc3.kl_loss


class VanillaMLP(nn.Module):
    def __init__(self, dropp=0):
        super(VanillaMLP, self).__init__()
        self._block = nn.Sequential(
            nn.Linear(784, 1200),
            nn.Dropout(dropp),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.Dropout(dropp),
            nn.ReLU(),
            nn.Linear(1200, 10)
        )
        self.to(DEVICE)

    def forward(self, x, dummy=None):
        return self._block(x)


lr = 1e-3
mo = 0.9

bb_mlp = BBMlp()
b_criterion = nn.CrossEntropyLoss(reduction='sum')
b_trainer = optim.SGD(bb_mlp.parameters(), lr, mo)

# 对比模型
vanilla_mlp = VanillaMLP()
dropout_mlp = VanillaMLP(dropp=0.5)
v_criterion = nn.CrossEntropyLoss()
v_trainer = optim.SGD(vanilla_mlp.parameters(), lr, mo)
d_criterion = nn.CrossEntropyLoss()
d_trainer = optim.SGD(dropout_mlp.parameters(), lr, mo)

b_writer = SummaryWriter(log_dir='./runs/b/')
v_writer = SummaryWriter(log_dir='./runs/v/')
d_writer = SummaryWriter(log_dir='./runs/d/')

n_epochs = 600
n_samples = 3

for e in range(n_epochs):
    bb_mlp.train()
    n_batchs = len(mnist_iter)
    for i, (x, y) in enumerate(mnist_iter):
        x = x.view(x.size(0), -1).to(DEVICE)
        y = y.to(DEVICE)

        # train
        rec_loss, kl_loss = 0, 0
        for _ in range(n_samples):
            rec_loss += b_criterion(bb_mlp(x), y)
            # reweight kl_loss
            # kl_reweight = 2 ** (n_batchs - i - 1) / (2 ** n_batchs - 1)
            # kl_loss = kl_reweight * bb_mlp.kl_loss()
            kl_loss += bb_mlp.kl_loss() / n_batchs
        b_loss = (rec_loss + kl_loss) / n_samples

        b_trainer.zero_grad()
        b_loss.backward()
        b_trainer.step()

        # 对比模型的训练
        v_loss = v_criterion(vanilla_mlp(x), y)
        v_trainer.zero_grad()
        v_loss.backward()
        v_trainer.step()

        d_loss = v_criterion(dropout_mlp(x), y)
        d_trainer.zero_grad()
        d_loss.backward()
        d_trainer.step()

        if i % 100 == 0:
            print('[Epoch: %d] [Batch: %d] rec_loss: %.3f kl_loss: %.3f b_loss: %.3f v_loss: %.3f d_loss: %.3f' % (
                e, i, rec_loss.item(), kl_loss.item(), b_loss.item(), v_loss.item(), d_loss.item()))

    # test
    acc = test(bb_mlp, test_iter)
    b_writer.add_scalar('test_acc', acc, e)
    acc = test(vanilla_mlp, test_iter)
    v_writer.add_scalar('test_acc', acc, e)
    acc = test(dropout_mlp, test_iter)
    d_writer.add_scalar('test_acc', acc, e)
