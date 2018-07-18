import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.nn.parameter import Parameter

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

EPS = 1e-10


class additive_coupling_layer(nn.Module):
    def __init__(self, in_features, hidden_dim, reverse=True):
        super(additive_coupling_layer, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.reverse = reverse
        self.permutation = torch.range(self.in_features - 1, 0, -1, dtype=torch.long)
        assert in_features % 2 == 0
        self.split = in_features // 2
        self.m = nn.Sequential(
            nn.Linear(self.split, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.split)
        )

    def forward(self, x, inv=False):
        if not inv:
            right = x[:, self.split:]
            left = self.m(x[:, self.split:]) + x[:, :self.split]
        else:
            right = x[:, self.split:]
            left = x[:, :self.split] - self.m(x[:, self.split:])
        if self.reverse:
            # return torch.cat([right, left], 1)  # 交换会导致生成图像错乱？？？
            return torch.cat([left, right], 1)[:, self.permutation]
        else:
            return torch.cat([left, right], 1)


class nice(nn.Module):
    def __init__(self, im_size):
        super(nice, self).__init__()
        self.cp1 = additive_coupling_layer(im_size, 1000)
        self.cp2 = additive_coupling_layer(im_size, 1000)
        self.cp3 = additive_coupling_layer(im_size, 1000)
        self.cp4 = additive_coupling_layer(im_size, 1000)
        self.scale = Parameter(torch.zeros(im_size))
        self.to(DEVICE)

    def forward(self, x, inv=False):
        if not inv:
            cp1 = self.cp1(x)
            cp2 = self.cp2(cp1)
            cp3 = self.cp3(cp2)
            cp4 = self.cp4(cp3)
            return torch.exp(self.scale) * cp4
        else:
            cp4 = x * torch.exp(-self.scale)
            cp3 = self.cp4(cp4, True)
            cp2 = self.cp3(cp3, True)
            cp1 = self.cp2(cp2, True)
            return self.cp1(cp1, True)

    def log_logistic(self, h):
        return -(F.softplus(h) + F.softplus(-h))

    def train_loss(self, h):
        return -(self.log_logistic(h).sum(1).mean() + self.scale.sum())  # + 0.1 * torch.abs(self.scale).sum()


# train
model = nice(784)
mnist_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
trainer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.5, 0.99])
lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=2, gamma=0.9)

n_epochs = 300
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(n_epochs):
    model.train()
    lr_scheduler.step()
    for batch_idx, (x, _) in enumerate(mnist_iter):
        x = x.view(x.size(0), -1).to(DEVICE)
        h = model(x)

        loss = model.train_loss(h)
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if batch_idx % 50 == 0:
            print('[ %d / %d ] loss: %.4f' % (epoch, n_epochs, loss.item()))

    with torch.no_grad():
        model.eval()
        # 从logistics中采样
        batch_size = 64
        z = torch.rand(batch_size, 784).to(DEVICE)
        h = torch.log(z + EPS) - torch.log(1 - z)
        x = model(h, inv=True)
        tv.utils.save_image(x.view(batch_size, 1, 28, 28), save_dir + 'nice_%d.png' % epoch)
