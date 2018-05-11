import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.optim import lr_scheduler

from utils import get_cls_accuracy

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

lr = 1e-3
n_epochs = 20

save_dir = './mutli_label_gan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 100
epoch_lr_decay = n_epochs / 2
save_epoch_freq = 2

dim_z = 16
dim_l = 11
dim_im = 784

batch_size = 64

train_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

test_iter = T.utils.data.DataLoader(
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

G = T.nn.Sequential(
    nn.Linear(dim_z, 256),
    nn.ReLU(),
    nn.Linear(256, dim_im),
    nn.ReLU()
).to(DEVICE)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.fc1 = T.nn.Sequential(
            nn.Linear(dim_im, 64),
            nn.ReLU())
        self.fc2 = T.nn.Sequential(
            nn.Linear(64, dim_l),
            nn.LogSoftmax())
        self.fc3 = T.nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, input):
        branch = self.fc1(input)
        return self.fc2(branch), self.fc3(branch)


D = Dis().to(DEVICE)

opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.99])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.99])
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=2, gamma=0.8)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=2, gamma=0.8)

G.train()
D.train()


def gaussian(size, mean=0, std=1):
    return T.normal(T.ones(size) * mean, std)


# 100 labeled data
X, L = next(iter(test_iter))
X = X.view(-1, dim_im).to(DEVICE)
L = L.to(DEVICE)

# train
for epoch in range(n_epochs):
    if epoch > epoch_lr_decay:
        scheduler_G.step()
        scheduler_D.step()

    _batch = 0
    for x, label in train_iter:
        _batch += 1
        # G
        x_real = x.view(-1, dim_im).to(DEVICE)
        label = label.to(DEVICE)
        z = gaussian([batch_size, dim_z]).to(DEVICE)

        fake_x = G(z)
        _, fake_score = D(fake_x)

        loss_G = T.mean(T.log(T.ones_like(fake_score) - fake_score + 1e-10))
        # loss_G = -T.mean(T.log(fake_score))  # 相比较上面的loss， 这个收敛的更快
        fake_label = 10 * T.ones(fake_x.size(0), dtype=T.int64).to(DEVICE)
        # loss_G -= F.nll_loss(fake_cl, fake_label)  # 添加此项，会破坏分类器的学习

        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # D
        fake_cl, fake_score = D(fake_x.detach())
        real_cl, real_score = D(x_real)
        X_cl, _ = D(X)

        loss_D = - T.mean(T.log(T.ones_like(fake_score) - fake_score + 1e-10) +
                          T.log(real_score + 1e-10))
        loss_D += F.nll_loss(X_cl, L) + F.nll_loss(fake_cl, fake_label)

        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        if _batch % print_every == 0:
            acc = get_cls_accuracy(real_cl.detach(), label)
            print('Epoch %d Batch %d ' % (epoch, _batch) +
                  'Loss D: %0.3f ' % loss_D.item() +
                  'Loss G: %0.3f ' % loss_G.item() +
                  'acc: %0.3f ' % acc.item() +
                  'F-score/R-score: [ %0.3f / %0.3f ]' %
                  (T.mean(fake_score).item(), T.mean(real_score).item()))

            _imags = fake_x.view(batch_size, 1, 28, 28).detach()
            tv.utils.save_image(_imags, save_dir + '{}_{}.png'.format(epoch, _batch))

# 查看训练集的分类准确率
total_acc = 0
b = 0
for x, label in train_iter:
    with T.no_grad():
        D.eval()
        preds, _ = D(x.view(-1, 784).to(DEVICE))
        total_acc += get_cls_accuracy(preds, label.to(DEVICE))
        b += 1
total_acc = total_acc / b
print('半监督的gan分类器的分类正确率：{}'.format(total_acc))

# 训练相同的分类器
classifier = T.nn.Sequential(
    nn.Linear(dim_im, 64),
    nn.ReLU(),
    nn.Linear(64, dim_l),
    nn.LogSoftmax()).to(DEVICE)

opt_C = optim.Adam(classifier.parameters(), lr, betas=[0.5, 0.99])
scheduler_C = lr_scheduler.StepLR(opt_C, step_size=2, gamma=0.8)

for epoch in range(n_epochs):
    if epoch > epoch_lr_decay:
        scheduler_C.step()

    logits = classifier(X)
    loss = F.nll_loss(logits, L)
    classifier.zero_grad()
    loss.backward()
    opt_C.step()

    print('Epoch %d  loss: %0.3f' % (epoch, loss.item()))

# test
total_acc = 0
b = 0
for x, label in train_iter:
    with T.no_grad():
        D.eval()
        preds = classifier(x.view(-1, 784).to(DEVICE))
        total_acc += get_cls_accuracy(preds, label.to(DEVICE))
        b += 1
total_acc = total_acc / b
print('普通分类器的分类正确率：{}'.format(total_acc))
