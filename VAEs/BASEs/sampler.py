# ref https://github.com/musyoku/adversarial-autoencoder/blob/master/aae/sampler.py
import random
from math import *

import numpy as np


def uniform(bz, ndim, minv=-1, maxv=1):
    return np.random.uniform(minv, maxv, (bz, ndim)).astype(np.float32)


def gaussian(bz, ndim, mu=0, var=1):
    return np.random.normal(mu, var, (bz, ndim)).astype(np.float32)
