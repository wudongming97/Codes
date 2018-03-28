import math

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from config import FLAGS

mnist = input_data.read_data_sets('./data/mnist')
fashion = input_data.read_data_sets('./data/fashion')


def next_batch_(bz, ds='mnist'):
    ds_ = mnist if ds == 'mnist' else fashion
    data = ds_.train.next_batch(bz, shuffle=True)
    return np.reshape(data[0], [bz, 28, 28, 1]), data[1]


def z_real_(bz, z_dim):
    return np.random.normal(0, 1, [bz, z_dim]) * 1.8


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


def embedding_viz_(x, y, ti):
    from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA

    x_tsne = TSNE().fit_transform(x)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
    plt.savefig(FLAGS.log_path + 'z_tsne_{}.png'.format(ti))
    # x_pca = PCA(2).fit_transform(x)
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
    # plt.savefig(FLAGS.log_path + 'z_pca_{}.png'.format(ti))
    plt.close()
