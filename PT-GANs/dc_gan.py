import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.optim import lr_scheduler

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

lr = 2e-4
n_epochs = 30

save_dir = './results_dcgan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 50
save_epoch_freq = 1

nz = 100
nc = 3
ndf = 128
ngf = 128
im_size = [64, 64]
batch_size = 128

_transformer = tv.transforms.Compose([
    tv.transforms.Resize(im_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
])

train_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.LSUN(
        root='../../Datasets/LSUN/',
        transform=_transformer,
        classes=['bedroom_train']
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


# train_iter = T.utils.data.DataLoader(
#     dataset=tv.datasets.CIFAR100(
#         root='../../Datasets/CIFAR100/',
#         transform=_transformer,
#         download=True,
#         train=True,
#     ),
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2,
# )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class generator(nn.Module):
    """
    dc_gan for lsun datasets
    """

    def __init__(self, nz, nc, ngf):
        super(generator, self).__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            # layer 2 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # layer 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # layer 4
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # layer 5
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return nn.parallel.data_parallel(self.net, z)


class discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # layer 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # layer 3
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # layer 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return nn.parallel.data_parallel(self.net, x).view(-1, 1).squeeze(1)


G = generator(nz, nc, ngf).apply(weights_init).to(DEVICE)
D = discriminator(3, ndf).apply(weights_init).to(DEVICE)

print('网络结构!!' + '\n' + '--' * 30)
print(G)
print(D)

opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.999])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.999])

scheduler_lr = lr_scheduler.StepLR(opt_G, step_size=1, gamma=0.9)

G.train()
D.train()

real_label = 1
fake_label = 0

# train
for epoch in range(0, n_epochs):
    print('训练：%d' % epoch + '--' * 30)
    _batch = 0
    scheduler_lr.step()
    for X, _ in train_iter:
        _batch += 1

        x_real = X.to(DEVICE)
        z = T.randn(x_real.size(0), nz, 1, 1, device=DEVICE)

        fake_x = G(z)
        fake_score = D(fake_x)
        real_score = D(x_real)

        r_label = T.full((batch_size,), real_label, device=DEVICE)
        f_label = T.full((batch_size,), fake_label, device=DEVICE)

        D.zero_grad()
        lss_D = F.binary_cross_entropy(real_score, r_label) + F.binary_cross_entropy(fake_score, f_label)
        lss_D.backward()
        opt_D.step()

        fake_score = D(fake_x.detach())
        real_score = D(x_real)

        G.zero_grad()
        lss_G = F.binary_cross_entropy(fake_score, r_label)
        lss_G.backward()
        opt_G.step()

        if _batch % print_every == 0:
            print('[%2d/%2d] [%5d/%5d] ' % (epoch, n_epochs, _batch, len(train_iter)) +
                  'loss_D: %0.3f ' % lss_D.item() +
                  'loss_G: %0.3f ' % lss_G.item() +
                  'F-score/R-score: [%0.3f/%0.3f]' %
                  (T.mean(fake_score).item(), T.mean(real_score).item()))

            tv.utils.save_image(fake_x.detach()[:64] * 0.5 + 0.5, save_dir + '{}_{}.png'.format(epoch, _batch))

    if epoch % save_epoch_freq == 0:
        T.save(D.state_dict(), 'dcgan_netd.pth')
        T.save(G.state_dict(), 'dcgan_netg.pth')
