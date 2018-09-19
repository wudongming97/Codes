import math

import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv

seed = 77
torch.manual_seed(seed)

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')


def log_gaussian(x, mu, sigma):
    return -0.5 * math.log(2.0 * np.pi) - math.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)


def lr_sgld_scheduler(t, init_lr=0.01, end_lr=0.0001, factor=0.55, total_steps=10000):
    # ref https://github.com/apache/incubator-mxnet/blob/master/example/bayesian-methods/sgld.ipynb
    b = (total_steps - 1.0) / ((init_lr / end_lr) ** (1.0 / factor) - 1.0)
    a = init_lr / (b ** (-factor))
    return a * (b + t) ** (-factor)


def lr_linear_scheduler(t, init_lr=0.01, end_lr=0.0001, total_steps=10000):
    return init_lr - t * (init_lr - end_lr) / total_steps


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = torch.max(score, dim=1)
    correct = torch.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy


def mnist_loaders(root, batch_size=128):
    trans = tv.transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=True, transform=trans, download=True),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=False, transform=trans),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SingleTensorDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size(0)


def tensor_loader(data, batch_size=1):
    data_loader = torch.utils.data.DataLoader(
        dataset=SingleTensorDataset(data), batch_size=batch_size, shuffle=True)
    return data_loader
