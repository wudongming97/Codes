from itertools import count

import gym
import numpy as np
import torch.optim as optim

from models import NaiveAC
from parallel_env import SubprocVecEnv
from utils import *

NUM_ENVS = 16
STEP_TO_TRAIN = 5
ENTROPY_BETA = 0.001

model_name = 'PG_A2C_01'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name


def make_env():
    def _thunk():
        env = gym.make(env_id)
        return env

    return _thunk


test_env = gym.make(env_id)
envs = [make_env() for i in range(NUM_ENVS)]
envs = SubprocVecEnv(envs)
net = NaiveAC(envs.observation_space.shape[0], envs.action_space.n)


def calc_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * masks[t]
        returns.insert(0, R)
    return returns


def n_step(num_steps, state):
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy_list = []
    for _ in range(num_steps):
        dist, val = net(state)
        action = dist.sample().squeeze()
        state, reward, done, _ = envs.step(action.cpu().numpy())
        values.append(val)
        log_probs.append(dist.log_prob(action).unsqueeze(1))
        entropy_list.append(dist.entropy().unsqueeze(1))
        rewards.append(torch.tensor(reward).float().unsqueeze(1).to(DEVICE))
        masks.append(torch.tensor(1 - done).float().unsqueeze(1).to(DEVICE))
    _, next_val = net(state)
    returns = calc_returns(next_val, rewards, masks)

    log_probs = torch.cat(log_probs, 1)
    returns = torch.cat(returns, 1).detach()
    values = torch.cat(values, 1)
    advantage = returns - values
    entropy_v = torch.cat(entropy_list, 1)

    return advantage.squeeze(), log_probs, entropy_v, state


state = envs.reset()
trainer = optim.Adam(net.parameters(), lr=5e-4, betas=[0.5, 0.999])

for idx in count(1):
    advantage, log_probs, entropy_v, state = n_step(STEP_TO_TRAIN, state)
    a_loss = -(log_probs * advantage.detach()).sum(1).mean()
    c_loss = 0.5 * advantage.pow(2).sum(1).mean()
    e_loss = ENTROPY_BETA * entropy_v.sum(1).mean()
    loss = a_loss + c_loss - e_loss

    trainer.zero_grad()
    loss.backward()
    trainer.step()

    if idx % 200 == 0:
        mean_reward = np.mean([test_policy(net, test_env) for _ in range(10)])
        print('Frame_idx: %d, loss: %.3f, a_loss: %.3f, c_loss: %.3f,  e_loss: %.3f, mean_reward: %.3f' % (
            idx, loss.item(), a_loss.item(), c_loss.item(), e_loss.item(), float(mean_reward)))
        if mean_reward >= 192:
            torch.save(net.state_dict(), identity + '.pth')
            print("Solved!")
            break
