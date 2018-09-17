import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

batch_size = 64


def plot_q_z(x, y, filename):
    from sklearn.manifold import TSNE
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]

    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    if x.shape[1] != 2:
        x = TSNE().fit_transform(x)
    y = y[:, np.newaxis]
    xy = np.concatenate((x, y), axis=1)

    for l, c in zip(range(10), colors):
        ix = np.where(xy[:, 2] == l)
        ax.scatter(xy[ix, 0], xy[ix, 1], c=c, marker='o', label=l, s=10, linewidths=0)

    plt.savefig(filename)
    plt.close()


def _file_paths(dir, ext):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith(ext):
            paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path, ext))
    return paths


class MonolingualImageFolder(data.Dataset):
    def __init__(self, root, ext='.png', transform=None):
        self.img_paths = _file_paths(root, ext)
        self.L = len(self.img_paths)
        self.transform = transform

    def __len__(self):
        return self.L

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        if self.transform is not None:
            img = self.transform(img)
        return img


def anime_face_loader(root, transform, batch_size=128):
    face_loader = torch.utils.data.DataLoader(
        dataset=MonolingualImageFolder(root=root, transform=transform),
        batch_size=batch_size, shuffle=True)
    return face_loader


def mnist_loaders(root, batch_size=128):
    trans = tv.transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=True, transform=trans, download=True),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=False, transform=trans),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
