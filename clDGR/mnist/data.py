import math
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist')
fashion = input_data.read_data_sets('./data/fashion')

pairs = sorted(zip(mnist.train.labels, mnist.train.images), key=lambda x: x[0])
pairs.extend(sorted(zip(fashion.train.labels, fashion.train.images), key=lambda x: x[0]))


def imcomb_(images):
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
    plt.imshow(np.squeeze(img))


def one_hot_(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


# print(len(pairs)) # 110000
class dataset(object):
    def __init__(self):
        self.num = len(pairs)
        self.start = 0

    def next_bt(self, bz=100, shuffle=False):
        self.start = 0 if self.start + bz > self.num else self.start + bz
        slice_ = random.sample(pairs, bz) if shuffle else pairs[self.start: self.start + bz]
        labels, images = list(zip(*slice_))
        return np.reshape(np.array(images), [bz, 28, 28, 1]), one_hot_(np.array(labels), 10)
