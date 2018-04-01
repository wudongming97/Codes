import math

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


def imcombind_(images, width=6):
    num = images.shape[0]
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:4]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, :]
    return image


def imsave_(path, img):
    plt.imsave(path, np.squeeze(img))


def inverse_transform(images):
    return (images + 1.) / 2.


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def L1Loss(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def L2Loss(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sceLoss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
