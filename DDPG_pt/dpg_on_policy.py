from itertools import count

import gym
import numpy as np
import torch.optim as optim

from env_wrappers import ActionNormalizedEnv
from models import *
from parallel_env import SubprocVecEnv
from utils import *

GAMMA = 0.99
NUM_ENVS = 16
STEP_TO_TRAIN = 5

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

model_name = 'dpg_on_policy'
env_id = "Pendulum-v0"
identity = env_id + '_' + model_name


def make_env():
    def _thunk():
        env = gym.make(env_id)
        return ActionNormalizedEnv(env)

    return _thunk


test_env = make_env()()
envs = [make_env() for i in range(NUM_ENVS)]
envs = SubprocVecEnv(envs)

obs_size = envs.observation_space.shape[0]
act_size = envs.action_space.shape[0]
act_net = DDPG_Actor(obs_size, act_size)
cri_net = DDPG_Critic(obs_size, act_size)


def calc_qval(next_value, rewards, masks, gamma=0.99):
    qval = next_value
    for t in reversed(range(len(rewards))):
        qval = rewards[t] + gamma * qval * masks[t]
    return qval


mse = nn.MSELoss()
act_trainer = optim.Adam(act_net.parameters(), lr=1e-4)
cri_trainer = optim.Adam(cri_net.parameters(), lr=1e-3)

state = envs.reset()

for frame_idx in count(1):
    qval = cri_net(to_tensor(state), act_net(to_tensor(state)))
    loss_act = -qval.sum(1).mean()

    rewards = []
    masks = []
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    for _ in range(STEP_TO_TRAIN):
        if np.random.uniform() > epsilon:
            action = act_net.get_action(state)
        else:
            action = np.random.uniform(size=[NUM_ENVS, 1])
        state, reward, done, _ = envs.step(action)
        rewards.append(torch.tensor(reward).float().unsqueeze(1).to(DEVICE))
        masks.append(torch.tensor(1 - done).float().unsqueeze(1).to(DEVICE))

    next_qval = cri_net(to_tensor(state), act_net(to_tensor(state))).detach()
    expected_qval = calc_qval(next_qval, rewards, masks, GAMMA)
    loss_cri = mse(qval, expected_qval.detach())

    cri_trainer.zero_grad()
    loss_cri.backward(retain_graph=True)
    cri_trainer.step()
    act_trainer.zero_grad()
    loss_act.backward()
    act_trainer.step()

    if frame_idx % 200 == 0:
        mean_reward = test_policy(act_net, test_env, n_episodes=10)
        print('Frame_idx: %d, loss_critic: %.3f, loss_actor: %.3f, mean_reward: %.3f' % (
            frame_idx, loss_cri.item(), loss_act.item(), float(mean_reward)))
