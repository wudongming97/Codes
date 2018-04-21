import os
from queue import Queue
from threading import Thread

import numpy as np
from matplotlib.image import imread


def datasets_info(name):
    return {
        'cat': './datasets/cats_with_mask/',
        'dog': './datasets/dogs_with_mask/'
    }[name]


def get_paired_paths(dir, end_with='.png'):
    paths = []
    for img in os.scandir(dir):
        if img.name.endswith(end_with):
            paths.append(img.path)

    paths.sort()
    i, pairs_, L = 0, [], len(paths)
    while i < L:
        pairs_.append((paths[i], paths[i + 1]))
        i += 2
    return pairs_


class datasets:
    def __init__(self, dir, capacity=600):
        self.paired_paths = get_paired_paths(dir)
        self.L = len(self.paired_paths)
        self.Q = Queue(maxsize=capacity)
        self.p_thread = Thread(target=self.producer)
        self.p_thread.start()

    def producer(self):
        while True:
            pair = self.paired_paths[np.random.choice(self.L)]
            image = np.reshape(imread(pair[0]) * 2 - 1, [1, 256, 256, 3])
            mask = np.reshape(imread(pair[1]), [1, 256, 256, 1])
            self.Q.put((image, mask))

    def shuffle_pair(self):
        pair = self.Q.get()
        return pair
