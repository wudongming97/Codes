import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./data/mnist')



class data_(object):

    @staticmethod
    def next_batch_(batch_sz):
        while True:
            yield np.reshape(mnist.train.next_batch(batch_sz)[0], [batch_sz, 28, 28, 1])

    @staticmethod
    def z_sample_(bz, z_dim):
       return np.random.normal(0, 1, [bz, z_dim])

    @staticmethod
    def imcombind_(images):
        num = images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = images.shape[1:4]
        image = np.zeros((height*shape[0], width*shape[1], shape[2]), dtype=images.dtype)
        for index, img in enumerate(images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, :]
        return image

    @staticmethod
    def imsave_(path, img):
        plt.imsave(path, np.squeeze(img), cmap=plt.cm.gray)

    @staticmethod
    def implot_(img):
        plt.imshow(img)


