import random
from collections import deque

import numpy as np
import torch.nn as nn
import torch.optim as optim

from atari_wrappers import make_atari, wrap_deepmind, ImageToPyTorch
from utils import *

# from tensorboardX import SummaryWriter

GAMMA = 0.99
N_FRAMES = 1400000
BATCH_SIZE = 32
REPLAY_SIZE = 10000
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


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


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                state = torch.tensor(state).unsqueeze(0).to(DEVICE)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
            else:
                action = random.randrange(env.action_space.n)
            return action


def calc_td_loss(batch, net, tgt_net):
    state, action, reward, next_state, done = to_tensor(batch)

    q_values = net(state)
    next_q_values = tgt_net(next_state).detach()
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + GAMMA * next_q_value * (1 - done.float())

    return nn.MSELoss()(q_value, expected_q_value)


env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = ImageToPyTorch(env)

replay_buffer = ReplayBuffer(REPLAY_SIZE)
net = DQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
trainer = optim.Adam(net.parameters(), lr=1e-5, betas=[0.5, 0.99])


# writer = SummaryWriter(comment='-' + env_id)


# random play to fill replay buffer
def explore_env(env, replay_buffer):
    state = env.reset()
    while True:
        action = random.randrange(env.action_space.n)
        next_state, reward, is_done, _ = env.step(action)
        replay_buffer.append(state, action, reward, next_state, is_done)
        state = next_state
        if is_done:
            return


while len(replay_buffer) < REPLAY_SIZE:
    explore_env(env, replay_buffer)

episode_reward = 0
total_rewards = []
state = env.reset()
best_mean_reward = None

for frame_idx in range(N_FRAMES):
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    action = net.act(state, epsilon)
    next_state, reward, is_done, _ = env.step(action)
    replay_buffer.append(state, action, reward, next_state, is_done)

    state = next_state
    episode_reward += reward

    if is_done:
        state = env.reset()
        total_rewards.append(episode_reward)
        episode_reward = 0

    # update
    loss = calc_td_loss(replay_buffer.sample(BATCH_SIZE), net, tgt_net)
    trainer.zero_grad()
    loss.backward()
    trainer.step()

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())
        mean_reward = np.mean(total_rewards[-1000:])
        if best_mean_reward is None or best_mean_reward < mean_reward:
            best_mean_reward = mean_reward
            torch.save(net.state_dict(), env_id + "-best.pth")
        print("frame_idx: %d loss: %.3f mean_reward: %.3f" % (frame_idx, loss.item(), mean_reward))
