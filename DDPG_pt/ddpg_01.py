from collections import deque

import gym
import numpy as np
import torch.optim as optim

from env_wrappers import ActionNormalizedEnv
from models import *
from ou import OUNoise
from replay_buffer import ReplayBuffer
from utils import *

GAMMA = 0.99
SOFT_TAU = 1e-2

model_name = 'ddpg_01'
env_id = "Pendulum-v0"
identity = model_name + '_' + env_id
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
    loss_act = -cri_net(state, act_net(state)).sum(1).mean()
    qval = cri_net(state, action)
    t_qval = cri_net_t(next_state, act_net_t(next_state))
    expected_qval = reward.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * GAMMA * t_qval
    loss_cri = mse_criterion(qval, expected_qval.detach())

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
batch_size = 64
frame_idx = 0
latest_10_returns = deque(maxlen=10)

while True:
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
        frame_idx += 1
        if len(replay_buffer) > batch_size:
            loss_act, loss_cri = ddpg_update(batch_size)
        episode_reward += reward
        if done:
            break
    latest_10_returns.append(episode_reward)
    mean_return = np.mean(latest_10_returns)
    if frame_idx % 500 == 0:
        print('Frame_idx: %d, loss_act: %.3f, loss_cri： %.3f, mean_return: %.3f' % (
            frame_idx, loss_act, loss_cri, float(mean_return)))
    if mean_return > -300:
        torch.save(act_net.state_dict(), identity + '_act.pth')
        torch.save(act_net.state_dict(), identity + '_cri.pth')
        print('Solved!')
        break
