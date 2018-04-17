import math

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torchvision

USE_GPU = T.cuda.is_available()
Tensor = T.cuda.FloatTensor if USE_GPU else T.FloatTensor

mnist_train = torchvision.datasets.MNIST(
    './datas/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

mnist_test = torchvision.datasets.MNIST(
    './datas/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)

fashion = torchvision.datasets.FashionMNIST(
    './datas/fashion',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

fashion_test = torchvision.datasets.FashionMNIST(
    './datas/fashion',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)


def next_batch(X, bs):
    end, L = bs, len(X)
    while end < L:
        batch_X = [X[i][0] for i in range(end - bs, end)]
        batch_X = T.cat(batch_X)
        if USE_GPU:
            batch_X = batch_X.cuda()
        yield batch_X
        end += bs


def shuffle_batch(X, bs):
    L = len(X)
    assert bs < L
    batch_X = [X[i][0] for i in np.random.choice(L, bs)]
    batch_X = T.cat(batch_X)
    if USE_GPU:
        batch_X = batch_X.cuda()
    return batch_X


def imcombind_(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, :]
    return image


def imsave_(path, img):
    plt.imsave(path, np.squeeze(img), cmap=plt.cm.gray)


def implot_(img):
    plt.imshow(img)


def to_gif(file_lst, gif_name):
    frames = [imageio.imread(file) for file in file_lst]
    imageio.mimsave(gif_name, frames, duration=0.5)


if __name__ == '__main__':
    # for a in next_batch(mnist, 1000):
    #     print(a.size())
    print(shuffle_batch(mnist_train, 1000).size())
