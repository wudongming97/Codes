import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.optim import lr_scheduler

from utils import *

lr = 2e-5
n_epochs = 8

batch_size = 32

save_dir = './ssl_gan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 500
epoch_lr_decay = n_epochs / 2
save_epoch_freq = 2

dim_z = 16
dim_l = 10
dim_im = 784

train_iter = torch.utils.data.DataLoader(
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


def label_data_batch(bs, max_size=200):
    X = tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    )
    L = len(X)
    assert bs < L and max_size <= L
    batch = [X[i] for i in np.random.choice(max_size, bs)]
    batch_X, batch_L = list(zip(*batch))
    batch_X, batch_L = torch.stack(batch_X), torch.stack(batch_L)
    return batch_X, batch_L


G = nn.Sequential(
    nn.Linear(dim_z, 256),
    nn.ReLU(),
    nn.Linear(256, dim_im),
    nn.ReLU()
).to(DEVICE)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_im, 64)
        self.fc2 = nn.Linear(64, dim_l)
        self.to(DEVICE)

    def forward(self, x):
        fc1 = F.relu(self.fc1(x))
        fc2 = self.fc2(fc1)
        return fc1, fc2


D = Discriminator()

opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.99])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.99])
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=2, gamma=0.8)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=2, gamma=0.8)

cl_criterion = nn.CrossEntropyLoss()
fm_criterion = nn.MSELoss()

# 用minst的测试机当着有标签的数据， 训练集为不带标签的数据
for epoch in range(n_epochs):
    G.train()
    D.train()
    if epoch > epoch_lr_decay:
        scheduler_G.step()
        scheduler_D.step()

    _batch = 0
    for x, _ in train_iter:
        _batch += 1
        label_x, y = label_data_batch(batch_size)
        x = x.view(-1, dim_im).to(DEVICE)
        label_x = label_x.view(-1, dim_im).to(DEVICE)
        y = y.to(DEVICE)

        # train D
        z = torch.randn(x.size(0), dim_z, device=DEVICE)
        fake_x = G(z)

        lab_logit = D(label_x)[-1]
        unl_logit = D(x)[-1]
        fak_logit = D(fake_x.detach())[-1]

        logz_lab, logz_unl, logz_fak = log_sum_exp(lab_logit), log_sum_exp(unl_logit), log_sum_exp(fak_logit)
        real_score = torch.mean(torch.exp(logz_unl - F.softplus(logz_unl)))
        fake_score = torch.mean(torch.exp(logz_fak - F.softplus(logz_fak)))

        d_supervised_loss = cl_criterion(lab_logit, y)
        d_unsupervised_loss = - torch.mean(logz_unl - F.softplus(logz_unl) - F.softplus(logz_fak))
        d_loss = d_supervised_loss + d_unsupervised_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # train G
        # # 一般的adv_loss
        # fak_logit = D(fake_x)[-1]
        # logz_fak = log_sum_exp(fak_logit)
        # g_loss = -torch.mean(F.softplus(logz_fak))

        # feature match
        last_2_fake = D(fake_x)[-2]
        last_2_real = D(x)[-2]
        g_loss = fm_criterion(last_2_fake.mean(0), last_2_real.detach().mean(0))
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        if _batch % print_every == 0:
            acc = get_cls_accuracy(lab_logit, y)
            print('[%d/%d] [%d] g_loss: %.3f d_loss: %.3f real_score: %.3f fake_score: %.3f acc: %.3f' % (
                epoch + 1, n_epochs, _batch, g_loss.item(), d_loss.item(), real_score.item(), fake_score.item(),
                acc.item()))
            tv.utils.save_image(fake_x[:16].view(-1, 1, 28, 28), save_dir + '{}_{}.png'.format(epoch + 1, _batch))

    # 查看对无标签数据的分类准确率
    total_acc = 0
    for x, label in train_iter:
        with torch.no_grad():
            D.eval()
            preds = D(x.view(-1, 784).to(DEVICE))[-1]
            total_acc += get_cls_accuracy(preds, label.to(DEVICE))
    total_acc = total_acc / len(train_iter)
    print('半监督的gan分类器的分类正确率：{}'.format(total_acc))

# 相同结构的一个baseline model 用于比较
baseline = Discriminator()
opt_b = optim.Adam(baseline.parameters(), lr, betas=[0.5, 0.99])
scheduler_b = lr_scheduler.StepLR(opt_G, step_size=2, gamma=0.8)

for epoch in range(n_epochs):
    if epoch > epoch_lr_decay:
        scheduler_b.step()
    baseline.train()
    _batch = 0
    for _ in train_iter:
        _batch += 1
        x, y = label_data_batch(batch_size)
        x = x.view(-1, dim_im).to(DEVICE)
        y = y.to(DEVICE)

        y_ = baseline(x)[-1]
        loss_b = cl_criterion(y_, y)

        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

        if _batch % print_every == 0:
            acc = get_cls_accuracy(y_, y)
            print('[%d/%d] [%d] loss: %.3f acc: %.3f' % (epoch + 1, n_epochs, _batch, loss_b.item(), acc.item()))

    # 查看对无标签数据的分类准确率
    total_acc = 0
    for x, label in train_iter:
        with torch.no_grad():
            baseline.eval()
            preds = baseline(x.view(-1, 784).to(DEVICE))[-1]
            total_acc += get_cls_accuracy(preds, label.to(DEVICE))
    total_acc = total_acc / len(train_iter)
    print('Baseline分类正确率：{}'.format(total_acc))
