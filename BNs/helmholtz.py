import os

import torch
import torchvision as tv
from torch.nn.functional import sigmoid

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

lr = 1e-2
n_epochs = 100
batch_size = 32

save_dir = './heim/'
os.makedirs(save_dir, exist_ok=True)

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

test_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

# net arch
h0_dim = 64
h1_dim = 256
v_dim = 784

# net weight init
B_g = torch.zeros(h0_dim, device=DEVICE)
W_g = torch.zeros(h0_dim, h1_dim, device=DEVICE)
V_g = torch.zeros(h1_dim, v_dim, device=DEVICE)
W_r = torch.zeros(h1_dim, h0_dim, device=DEVICE)
V_r = torch.zeros(v_dim, h1_dim, device=DEVICE)

for e in range(n_epochs):
    lr = lr * 0.9
    for x, _ in train_iter:
        x = x.view(-1, 28 * 28).to(DEVICE)
        x = (x > 0.5).float()
        # wake phase
        h1 = torch.bernoulli(sigmoid(x @ V_r))
        h0 = torch.bernoulli(sigmoid(h1 @ W_r))

        ksi = sigmoid(B_g)
        psi = sigmoid(h0 @ W_g)
        delta = sigmoid(h1 @ V_g)

        B_g += lr * torch.mean(h0 - ksi, 0)
        W_g += lr * h0.t() @ (h1 - psi) / batch_size
        V_g += lr * h1.t() @ (x - delta) / batch_size

        # sleep phase
        h0 = torch.bernoulli(sigmoid(B_g).repeat(batch_size, 1))
        h1 = torch.bernoulli(sigmoid(h0 @ W_g))
        v = torch.bernoulli(sigmoid(h1 @ V_g))

        psi = sigmoid(v @ V_r)
        ksi = sigmoid(h1 @ W_r)

        V_r += lr * v.t() @ (h1 - psi) / batch_size
        W_r += lr * h1.t() @ (h0 - ksi) / batch_size

    print('[%d/%d]' % (e + 1, n_epochs))

    # top-down 随机采样
    h0 = torch.bernoulli(sigmoid(B_g).repeat(64, 1))
    h1 = torch.bernoulli(sigmoid(h0 @ W_g))
    v = torch.bernoulli(sigmoid(h1 @ V_g))
    tv.utils.save_image(v.view(-1, 1, 28, 28), save_dir + 'r{}.png'.format(e + 1))

    # 重构
    x = next(iter(test_iter))[0]
    x = x.view(-1, 28 * 28).to(DEVICE)
    x = (x > 0.5).float()

    h1 = torch.bernoulli(sigmoid(x @ V_r))
    h0 = torch.bernoulli(sigmoid(h1 @ W_r))
    # tv.utils.save_image(h0.view(-1, 1, 8, 8), save_dir + 'h0_{}.png'.format(e + 1))
    h1 = torch.bernoulli(sigmoid(h0 @ W_g))
    v = torch.bernoulli(sigmoid(h1 @ V_g))
    tv.utils.save_image(x.view(-1, 1, 28, 28), save_dir + 'x{}.png'.format(e + 1))
    tv.utils.save_image(v.view(-1, 1, 28, 28), save_dir + 'v{}.png'.format(e + 1))

    # 显示filter
    tv.utils.save_image(V_r.t()[:64].view(-1, 1, 28, 28), save_dir + 'V_r{}.png'.format(e + 1))
    tv.utils.save_image(V_g.t()[:64].view(-1, 1, 16, 16), save_dir + 'V_g{}.png'.format(e + 1))
    tv.utils.save_image(W_r.t()[:64].view(-1, 1, 16, 16), save_dir + 'W_r{}.png'.format(e + 1))
    tv.utils.save_image(W_g.t()[:64].view(-1, 1, 8, 8), save_dir + 'W_g{}.png'.format(e + 1))
