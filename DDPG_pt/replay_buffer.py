import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state[np.newaxis, :], action, reward, next_state[np.newaxis, :], done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.vstack(state), action, reward, np.vstack(next_state), done

    def __len__(self):
        return len(self.buffer)
