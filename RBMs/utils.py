import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader as Data
import torchvision

USE_GPU = torch.cuda.is_available()

mnist = Data.DataLoader(
    dataset=torchvision.datasets.MNIST(
        './datas/mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=16, shuffle=True, num_workers=2)


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
