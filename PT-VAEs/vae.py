import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *

save_dirs = './results/'
os.makedirs(save_dirs, exist_ok=True)

im_dim = 784
z_dim = 16


class vae(nn.Module):
    def __init__(self):
        super(vae, self).__init__()

        self.E = nn.Sequential(
            nn.Linear(im_dim, 128),
            # nn.Dropout(p=0.3),
            nn.ReLU()
        )

        self.mu = nn.Linear(128, z_dim)
        self.log_var = nn.Linear(128, z_dim)

        self.D = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward_to_z(self, input):
        out = self.E(input)
        mu = self.mu(out)
        log_var = self.log_var(out)
        return mu, log_var

    def forward(self, z):
        return self.D(z)


def sample_z(mu, log_var):
    return mu + T.exp(0.5 * log_var) * T.randn(mu.size()).to(DEVICE)


def train(model, data_iter, lr=1e-3, n_epochs=10):
    trainer = optim.Adam(model.parameters(), lr, betas=[0.5, 0.99])

    for e in range(n_epochs):
        for b, (x, _) in enumerate(data_iter):
            model.train()
            x = x.view(batch_size, -1).to(DEVICE)
            mu, log_var = model.forward_to_z(x)
            kl_loss = T.mean(-0.5 * T.sum(log_var - mu ** 2 + 1. - T.exp(log_var), 1))
            z = sample_z(mu, log_var)
            rec_x = model(z)
            rec_loss = F.binary_cross_entropy(rec_x, x, size_average=False) / batch_size

            loss = kl_loss + rec_loss
            model.zero_grad()
            loss.backward()
            trainer.step()

            if b % 100 == 0:
                print('[ %d / %d ] kl_loss: %.4f rec_loss: %4f ' % (e, n_epochs, kl_loss.item(), rec_loss.item()))

        # test
        with T.no_grad():
            # 随机生成
            model.eval()
            z = T.randn(64, z_dim).to(DEVICE)
            test_ims = model(z).view(-1, 1, 28, 28)
            tv.utils.save_image(test_ims, save_dirs + 'im_{}.png'.format(e))

            # 查看z空间
            x, l = next(iter(test_iter))
            x = x.view(-1, im_dim).to(DEVICE)
            z = sample_z(*model.forward_to_z(x))
            plot_q_z(z, l, save_dirs + 'z_{}.png'.format(e))


if __name__ == '__main__':
    model = vae().to(DEVICE)
    train(model, train_iter, 1e-3, 20)
