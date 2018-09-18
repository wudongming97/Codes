import math
import os

import pyro
import pyro.distributions as dist_pr
import pyro.optim as optim_pr
import torch
import torch.nn as nn
import torchvision as tv
from pyro.infer import SVI, Trace_ELBO

from utils import DEVICE, anime_face_loader


class Conv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, use_bias=True, transpose=False,
                 act_cls=nn.ReLU):
        super(Conv2dBlock, self).__init__()
        conv_ops = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._block = nn.Sequential(
            conv_ops(input_nc, output_nc, kernel_size, stride, padding, bias=use_bias),
            nn.BatchNorm2d(output_nc),
            act_cls()
        )

    def forward(self, x):
        return self._block(x)


class Encoder(nn.Module):
    def __init__(self, nf, z_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self._conv = nn.Sequential(
            # 3 x 64 x 64
            Conv2dBlock(3, nf, 4, 2, 1),
            Conv2dBlock(nf, nf * 2, 4, 2, 1),
            Conv2dBlock(nf * 2, nf * 2, 4, 2, 1),  # 8x8
        )
        conv_size = nf * 2 * 8 * 8
        self._fc = nn.Sequential(
            nn.Linear(conv_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, z_dim * 2)
        )

    def forward(self, x):
        x = self._conv(x)
        x = x.view(x.size(0), -1)
        x = self._fc(x)
        return x[:, :self.z_dim], x[:, self.z_dim:].exp()


class Decoder(nn.Module):
    def __init__(self, nf, z_dim):
        super(Decoder, self).__init__()
        self.nf = nf
        conv_size = self.nf * 2 * 8 * 8
        self._fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, conv_size),
            nn.ReLU()
        )
        self._conv_t = nn.Sequential(
            Conv2dBlock(nf * 2, nf * 2, 4, 2, 1, transpose=True),
            Conv2dBlock(nf * 2, nf, 4, 2, 1, transpose=True),
            nn.ConvTranspose2d(nf, 6, 4, 2, 1),
        )

    def forward(self, x):
        x = self._fc(x)
        x = x.view(-1, self.nf * 2, 8, 8)
        x = self._conv_t(x)
        return x[:, :3, :, :], x[:, 3:, :, :].exp()


class VAE(nn.Module):
    def __init__(self, z_dim=128, nf=32):
        super(VAE, self).__init__()
        self.enc = Encoder(nf, z_dim)
        self.dec = Decoder(nf, z_dim)
        self.to(DEVICE)
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        pyro.module("dec", self.dec)
        with pyro.iarange("data", x.size(0)):
            z_loc = x.new_zeros(x.size(0), self.z_dim)
            z_scale = x.new_ones(x.size(0), self.z_dim)
            z = pyro.sample("latent", dist_pr.Normal(z_loc, z_scale).independent(1))
            x_loc, x_scale = self.dec(z)
            pyro.sample("obs", dist_pr.Normal(x_loc, x_scale).independent(3), obs=x)
            return x_loc, x_scale

    def guide(self, x):
        pyro.module("enc", self.enc)
        with pyro.iarange("data", x.size(0)):
            z_loc, z_scale = self.enc(x)
            pyro.sample("latent", dist_pr.Normal(z_loc, z_scale).independent(1))

    def z_to_samples(self, f_name, batch_size=64):
        with torch.no_grad():
            self.dec.eval()
            z = torch.randn(batch_size, self.z_dim).to(DEVICE)
            x_loc, x_scale = self.dec(z)
            images = torch.distributions.Normal(x_loc, x_scale).sample()
            images = torch.clamp(images, min=0., max=1.)
            tv.utils.save_image(images, f_name, int(math.sqrt(batch_size)))
            self.dec.train()  # 恢复


def train(n_epochs, save_dir):
    pyro.clear_param_store()

    # dataset
    _transformer = tv.transforms.Compose([
        tv.transforms.Resize([64, 64]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    face_loader = anime_face_loader("../../Datasets/动漫头像/", _transformer, batch_size=64)

    # model and optimizer
    vae = VAE()
    trainer = optim_pr.Adam({'lr': 0.001})
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, trainer, loss=elbo)

    for epoch in range(n_epochs):
        for i, (x, _) in enumerate(face_loader):
            x = x.to(DEVICE)
            loss = svi.step(x)

            if i % 100 == 0:
                print("[Epoch: %d] [Batch: %d] [Loss: %d]" % (epoch, i, loss))

        vae.z_to_samples(save_dir + '%d.png' % epoch, 64)


if __name__ == '__main__':
    save_dir = './results/'
    os.makedirs(save_dir, exist_ok=True)
    n_epochs = 100
    train(n_epochs, save_dir)
