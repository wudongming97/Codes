from collections import deque
from itertools import count

import gym
import torch.optim as optim

from models import *

LR = 0.0005
GAMMA = 0.99
ENTROPY_BETA = 0.001

model_name = 'PG_NaiveAC'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name
env = gym.make(env_id)
net = NaiveAC(env.observation_space.shape[0], env.action_space.n)


def calc_returns(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


#  蒙特卡洛方法，需要完整的episode，
def one_episode():
    rewards = []
    values = []
    selected_logprobs = []
    entropy_list = []

    state = env.reset()
    while True:
        action, log_prob, val, entropy = net.action_and_logprob(state)
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
        selected_logprobs.append(log_prob)
        entropy_list.append(entropy)
        values.append(val)
        if is_done:
            break

    return rewards, selected_logprobs, values, entropy_list


# train
last_100_rewards = deque(maxlen=100)
trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.999])
for i_episode in count(1):
    p_loss = 0.0
    v_loss = 0.0
    rewards, selected_logprobs, values, entropy_list = one_episode()
    returns = calc_returns(rewards)
    for ret, val, logprob, entropy in zip(returns, values, selected_logprobs, entropy_list):
        advantage = ret - val
        p_loss -= (advantage * logprob + ENTROPY_BETA * entropy)
        v_loss += F.smooth_l1_loss(val, torch.tensor([ret]).unsqueeze(0).to(DEVICE))
    loss = p_loss + v_loss
    trainer.zero_grad()
    loss.backward()
    trainer.step()

    last_100_rewards.append(sum(rewards))
    mean_reward = np.mean(last_100_rewards)
    if i_episode % 10 == 0:
        print('Episode: %d, loss: %.3f, p_loss: %.3f, v_loss: %.3f,  mean_reward: %.3f' % (
            i_episode, loss.item(), p_loss.item(), v_loss.item(), float(mean_reward)))

    # 停时条件
    if mean_reward >= 198:
        print("Solved!")
        torch.save(net.state_dict(), identity + '.pth')
        break
