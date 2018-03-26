import math
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist')
fashion = input_data.read_data_sets('./data/fashion')

Pairs = sorted(zip(mnist.train.labels, mnist.train.images), key=lambda x: x[0])
Pairs.extend(sorted(zip(fashion.train.labels, fashion.train.images), key=lambda x: x[0]))


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
        self.num = len(Pairs)
        self.start = 5000

    def next_bt(self, bz=100, shuffle=False):
        if self.start + bz > self.num:
            self.start = 0
        slice_ = random.sample(Pairs, bz) if shuffle else Pairs[self.start: self.start + bz]
        self.start = self.start + bz
        labels, images = list(zip(*slice_))
        return np.reshape(np.array(images), [bz, 28, 28, 1]), one_hot_(np.array(labels), 10)

    def pre_bt(self, bz=1000):
        if self.start >= bz:
            slice_ = Pairs[self.start - bz:self.start]
        else:
            slice_ = Pairs[:self.start]
        random.shuffle(slice_)
        labels, images = list(zip(*slice_))
        return np.reshape(np.array(images), [-1, 28, 28, 1]), one_hot_(np.array(labels), 10)


if __name__ == '__main__':
    datas = dataset().next_bt()
    imsave_('./images/test.png', imcomb_(datas[0]))