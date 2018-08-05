from collections import deque
from itertools import count

import gym
import torch.optim as optim

from models import *

LR = 0.0005
GAMMA = 0.99
CLIP_GRAD = 0.2
ENTROPY_BETA = 0.001
EPISODES_TO_TRAIN = 4

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
def n_episode(n):
    episode_rewards = []
    returns = []
    entropy_list = []
    selected_logprobs = []
    vals = []

    for _ in range(n):
        state = env.reset()
        rewards = []
        while True:
            dist, value = net(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            state, reward, is_done, _ = env.step(action.item())
            rewards.append(reward)
            vals.append(value)
            entropy_list.append(entropy)
            selected_logprobs.append(log_prob)
            if is_done:
                episode_rewards.append(sum(rewards))
                returns.extend(calc_returns(rewards))
                break

    return episode_rewards, returns, selected_logprobs, vals, entropy_list


# train
last_100_rewards = deque(maxlen=100)
trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.999])
for i_episode in count(1):
    p_loss = 0.0
    v_loss = 0.0
    episode_rewards, returns, selected_logprobs, vals, entropy_list = n_episode(EPISODES_TO_TRAIN)
    for ret, val, logprob, entropy in zip(returns, vals, selected_logprobs, entropy_list):
        advantage = ret - val
        p_loss -= (advantage * logprob + ENTROPY_BETA * entropy)
        v_loss += F.smooth_l1_loss(val.squeeze(), torch.tensor([ret]).to(DEVICE))
    loss = (p_loss + v_loss) / EPISODES_TO_TRAIN
    trainer.zero_grad()
    loss.backward()
    # nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    trainer.step()

    last_100_rewards.extend(episode_rewards)
    mean_reward = np.mean(last_100_rewards)
    if i_episode % 10 == 0:
        print('Episode: %d, loss: %.3f, p_loss: %.3f, v_loss: %.3f,  mean_reward: %.3f' % (
            i_episode, loss.item(), p_loss.item(), v_loss.item(), float(mean_reward)))

    # 停时条件
    if mean_reward >= 198:
        print("Solved!")
        torch.save(net.state_dict(), identity + '.pth')
        break
