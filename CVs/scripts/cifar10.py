import os
import pickle

import numpy as np
from matplotlib.image import imsave


def unpickle(file):
    with open(file, 'rb') as fo:
        dick = pickle.load(fo, encoding='bytes')
    return dick


if __name__ == '__main__':

    files_ = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    label_names = unpickle('batches.meta')[b'label_names']

    for file in files_:
        dick = unpickle(file)
        data, lables, filenames = dick[b'data'], dick[b'labels'], dick[b'filenames']

        for ix, l_ in enumerate(lables):
            image = np.reshape(np.reshape(data[ix], [3, 1024]).T, [32, 32, 3])
            save_path = os.path.join('raw/' + str(l_) + '_' + str(filenames[ix], encoding='utf-8'))
            imsave(save_path, image)
