from collections import deque
from itertools import count

import gym
import torch.optim as optim
from tensorboardX import SummaryWriter

from models import *

LR = 0.001
GAMMA = 0.99
ENTROPY_BETA = 0.0001

model_name = 'ReinforceNaive_01'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name
env = gym.make(env_id)
net = PolicyNet(env.observation_space.shape[0], env.action_space.n)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


def one_episode():
    rewards = []
    entropy_list = []
    selected_logprobs = []

    state = env.reset()
    while True:
        action, log_prob, entropy = net.action_and_logprob(state)
        state, reward, is_done, _ = env.step(action)
        rewards.append(reward)
        entropy_list.append(entropy)
        selected_logprobs.append(log_prob)
        if is_done:
            break

    return rewards, selected_logprobs, entropy_list


# train
last_100_rewards = deque(maxlen=100)
trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.999])
writer = SummaryWriter(comment=identity)

for i_episode in count(1):
    pg_loss = 0.0
    rewards, selected_logprobs, entropy_list = one_episode()
    qvals = calc_qvals(rewards)
    for qval, logprob in zip(qvals, selected_logprobs):
        pg_loss -= qval * logprob
    entropy_loss = ENTROPY_BETA * sum(entropy_list)
    loss = pg_loss - entropy_loss
    trainer.zero_grad()
    loss.backward()
    trainer.step()

    last_100_rewards.append(sum(rewards))
    mean_reward = np.mean(last_100_rewards)

    writer.add_scalar('mean_reward', mean_reward, i_episode)
    writer.add_scalar('entropy_loss', entropy_loss.item(), i_episode)
    writer.add_scalar('pg_loss', pg_loss.item(), i_episode)
    writer.add_scalar('loss', loss.item(), i_episode)

    if i_episode % 10 == 0:
        print('Episode: %d, loss: %.3f, mean_reward: %.3f' % (i_episode, loss.item(), float(mean_reward)))

    # 停时条件
    if mean_reward >= 198:
        print("Solved!")
        torch.save(net.state_dict(), identity + '.pth')
        break
