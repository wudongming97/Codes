# simple implementation of Adversarial Variational Bayes
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

save_dir = './avb/'
os.makedirs(save_dir, exist_ok=True)

dim_z = 32
dim_im = 784

T = nn.Sequential(
    nn.Linear(dim_im + dim_z, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 128),
    nn.ReLU(True),
    nn.Linear(128, 1)
).to(DEVICE)

E = nn.Sequential(
    nn.Linear(dim_im + dim_z, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.Linear(256, dim_z)
).to(DEVICE)

D = nn.Sequential(
    nn.Linear(dim_z, 256),
    nn.ReLU(True),
    nn.Linear(256, 1024),
    nn.ReLU(True),
    nn.Linear(1024, dim_im),
    nn.Sigmoid()
).to(DEVICE)

n_epochs = 10
lr = 2e-4

opt_t = optim.Adam(T.parameters(), lr, betas=[0.5, 0.999])
opt_e = optim.Adam(E.parameters(), lr, betas=[0.5, 0.999])
opt_d = optim.Adam(D.parameters(), lr, betas=[0.5, 0.999])

bce_criterion = nn.BCELoss()

fix_noise = torch.randn(64, dim_z, device=DEVICE)  # only for test

for e in range(n_epochs):
    T.train()
    D.train()
    E.train()
    for b, (x, _) in enumerate(train_iter):
        bs = x.size(0)

        x = x.view(-1, dim_im).to(DEVICE)
        eps = torch.randn(bs, dim_z, device=DEVICE)

        pz = torch.randn(bs, dim_z, device=DEVICE)
        qz = E(torch.cat([x, eps], 1))

        t_xpz = T(torch.cat([x, pz], 1))
        t_xqz = T(torch.cat([x, qz], 1))

        loss_t = -torch.mean(torch.log(F.sigmoid(t_xqz) + 1e-10) + torch.log(1 - F.sigmoid(t_xpz) + 1e-10))

        opt_t.zero_grad()
        loss_t.backward(retain_graph=True)
        opt_t.step()

        rec_x = D(qz)
        loss_d = bce_criterion(rec_x, x)

        opt_d.zero_grad()
        loss_d.backward(retain_graph=True)
        opt_d.step()

        rec_x = D(qz)
        loss_e = bce_criterion(rec_x, x) + torch.mean(t_xqz)

        opt_e.zero_grad()
        loss_e.backward()
        opt_e.step()

        if (b + 1) % 50 == 0:
            tv.utils.save_image(rec_x[:16].view(-1, 1, 28, 28), save_dir + 'rec_{}_{}.png'.format(e + 1, b + 1))
            print('[%d/%d] [%d] loss_t: %.3f loss_d: %.3f loss_e: %.3f t_xpz: %.3f t_xqz: %.3f' % (
                e + 1, n_epochs, b + 1, loss_t.item(), loss_d.item(), loss_e.item(), F.sigmoid(t_xpz).mean().item(),
                F.sigmoid(t_xqz).mean().item()
            ))

    with torch.no_grad():
        D.eval()
        E.eval()
        # 1. 查看水机生成
        ims = D(fix_noise).view(-1, 1, 28, 28)
        tv.utils.save_image(ims, save_dir + 'r_{}.png'.format(e + 1))
        # 2. 查看z空间
        x, l = next(iter(test_iter))
        x = x.view(-1, dim_im).to(DEVICE)
        eps = torch.randn(1000, dim_z, device=DEVICE)
        qz = E(torch.cat([x, eps], 1))
        plot_q_z(qz, l, save_dir + 'z_{}.png'.format(e + 1))
