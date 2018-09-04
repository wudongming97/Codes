import random
from collections import deque, namedtuple

import numpy as np

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')


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


# 一个简单的EpisodeBuffer，只针对没有显式episode结束的env，以固定长度为一个Episode的环境，如pendulum等
class EpisodeBuffer(object):
    def __init__(self, capacity, episode_length):
        self.episode_length = episode_length
        self.buffer = deque(maxlen=capacity)
        self.trajectory = []

    def append(self, state, action, reward, next_state, done):
        self.trajectory.append(
            Experience(state=state, action=action, reward=reward, next_state=next_state, done=done))
        if len(self.trajectory) >= self.episode_length:
            self.buffer.append(self.trajectory)
            self.trajectory = []

    def sample(self, batch_size, t):
        batch = [self.buffer[ix] for ix in random.sample(range(len(self.buffer)), batch_size)]
        experiences = list(map(list, zip(*batch)))
        state = np.stack((trajectory.state for trajectory in experiences[t]))
        action = np.stack((trajectory.action for trajectory in experiences[t]))
        reward = np.stack((trajectory.reward for trajectory in experiences[t]))
        next_state = np.stack((trajectory.next_state for trajectory in experiences[t]))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
