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
    def __init__(self, in_features):
        super(additive_coupling_layer, self).__init__()
        assert in_features % 2 == 0
        self.split = in_features // 2
        self.in_features = in_features
        self.m1 = nn.Sequential(
            nn.Linear(self.split, self.split),
            nn.ReLU()
        )
        self.m2 = nn.Sequential(
            nn.Linear(self.split, self.split),
            nn.ReLU()
        )
        self.m3 = nn.Sequential(
            nn.Linear(self.split, self.split),
            nn.ReLU()
        )
        self.m4 = nn.Sequential(
            nn.Linear(self.split, self.split),
            nn.ReLU()
        )

    def forward(self, x, inv=False):
        if not inv:
            h0_0 = x[:, :self.split]
            h0_1 = x[:, self.split:]
            h1_1 = h0_1
            h1_0 = self.m1(h0_1) + h0_0
            h2_0 = h1_0
            h2_1 = self.m2(h1_0) + h1_1
            h3_1 = h2_1
            h3_0 = self.m3(h2_1) + h2_0
            h4_0 = h3_0
            h4_1 = self.m4(h3_0) + h3_1

            return torch.cat([h4_0, h4_1], 1)
        else:
            h4_0 = x[:, :self.split]
            h4_1 = x[:, self.split:]
            h3_0 = h4_0
            h3_1 = h4_1 - self.m4(h4_0)
            h2_1 = h3_1
            h2_0 = h3_0 - self.m3(h3_1)
            h1_0 = h2_0
            h1_1 = h2_1 - self.m2(h2_0)
            h0_1 = h1_1
            h0_0 = h1_0 - self.m1(h1_1)
            return torch.cat([h0_0, h0_1], 1)


class nice(nn.Module):
    def __init__(self, im_size):
        super(nice, self).__init__()
        self.cp1 = additive_coupling_layer(im_size)
        self.cp2 = additive_coupling_layer(im_size)
        self.cp3 = additive_coupling_layer(im_size)
        self.cp4 = additive_coupling_layer(im_size)

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
        return -(self.log_logistic(h).sum(1).mean() + self.scale.sum())


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

n_epochs = 100
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(n_epochs):
    model.train()
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
        z = torch.rand(64, 784).to(DEVICE)
        h = torch.log(z + EPS) - torch.log(1 - z)
        x = model(h, True)
        tv.utils.save_image(x.view(64, 1, 28, 28), save_dir + '%d.png' % epoch)
