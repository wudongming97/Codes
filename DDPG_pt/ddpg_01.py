from collections import deque
from itertools import count

import gym
import numpy as np
import torch.optim as optim

from env_wrappers import ActionNormalizedEnv
from models import *
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from utils import *

GAMMA = 0.99
SOFT_TAU = 1e-2

env_id = "Pendulum-v0"
env = ActionNormalizedEnv(gym.make(env_id))
ou_noise = OUNoise(env.action_space)

obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

act_net = DDPG_Actor(obs_size, act_size)
cri_net = DDPG_Critic(obs_size, act_size)
act_net_t = DDPG_Actor(obs_size, act_size)
cri_net_t = DDPG_Critic(obs_size, act_size)

hard_update(act_net_t, act_net)
hard_update(cri_net_t, cri_net)

mse_criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(10000)
act_trainer = optim.Adam(act_net.parameters(), lr=1e-4)
cri_trainer = optim.Adam(cri_net.parameters(), lr=1e-3)


def ddpg_update(batch_size):
    state, action, reward, next_state, done = to_tensor(replay_buffer.sample(batch_size))
    value = cri_net(state, act_net(state))
    loss_act = - value.sum(1).mean()
    t_value = cri_net_t(next_state, act_net_t(next_state))
    expected_value = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * GAMMA * t_value
    loss_cri = mse_criterion(value, expected_value.detach())

    act_trainer.zero_grad()
    loss_act.backward(retain_graph=True)
    act_trainer.step()
    cri_trainer.zero_grad()
    loss_cri.backward()
    cri_trainer.step()

    soft_update(act_net_t, act_net, SOFT_TAU)
    soft_update(cri_net_t, cri_net, SOFT_TAU)

    return loss_act.item(), loss_cri.item()


# train
max_steps = 500
batch_size = 128
latest_100_returns = deque(maxlen=100)

for frame_idx in count(1):
    state = env.reset()
    ou_noise.reset()
    episode_reward = 0
    loss_act, loss_cri = 0, 0

    for t in range(max_steps):
        action = act_net.action(state)
        action = ou_noise.get_action(action, t)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append(state, action, reward, next_state, done)
        state = next_state
        if len(replay_buffer) > batch_size:
            loss_act, loss_cri = ddpg_update(batch_size)
        episode_reward += reward
        if done:
            break
    latest_100_returns.append(episode_reward)
    if frame_idx % 20 == 0:
        mean_return = np.mean(latest_100_returns)
        print('Frame_idx: %d, loss_act: %.3f, loss_cri： %.3f, mean_return: %.3f' % (
            frame_idx, loss_act, loss_cri, float(mean_return)))
