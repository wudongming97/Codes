from collections import deque
from itertools import count

import torch.optim as optim
from tensorboardX import SummaryWriter

from atari_wrappers import get_env
from models import *

LR = 0.0005
GAMMA = 0.99
ENTROPY_BETA = 0.0001
GRAD_L2_CLIP = 0.1
EPISODES_TO_TRAIN = 4

model_name = 'ReinforceNaive_02'
env_id = "PongNoFrameskip-v4"
identity = env_id + '_' + model_name
env = get_env(env_id)
net = AtariPolicyNet(env.observation_space.shape, env.action_space.n)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = sum(res) / len(res)  # baseline
    return [q - mean_q for q in res]


def n_episode(n):
    episode_rewards = []
    qvals = []
    entropy_list = []
    selected_logprobs = []

    for _ in range(n):
        state = env.reset()
        rewards = []
        while True:
            dist = net(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            state, reward, is_done, _ = env.step(action.item())
            rewards.append(float(reward))
            entropy_list.append(entropy)
            selected_logprobs.append(log_prob)
            if is_done:
                episode_rewards.append(sum(rewards))
                qvals.extend(calc_qvals(rewards))
                break

    return episode_rewards, qvals, selected_logprobs, entropy_list


# train
last_100_rewards = deque(maxlen=100)
trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.999])
writer = SummaryWriter(comment=identity)

for i_episode in count(1):
    pg_loss = 0.0
    episode_rewards, qvals, selected_logprobs, entropy_list = n_episode(EPISODES_TO_TRAIN)
    for qval, logprob in zip(qvals, selected_logprobs):
        pg_loss -= qval * logprob
    entropy_loss = ENTROPY_BETA * sum(entropy_list)
    loss = (pg_loss - entropy_loss) / EPISODES_TO_TRAIN
    trainer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
    trainer.step()

    last_100_rewards.extend(episode_rewards)
    mean_reward = np.mean(last_100_rewards)

    writer.add_scalar('mean_reward', mean_reward, i_episode)
    writer.add_scalar('entropy_loss', entropy_loss.item(), i_episode)
    writer.add_scalar('pg_loss', pg_loss.item(), i_episode)
    writer.add_scalar('loss', loss.item(), i_episode)

    if i_episode % 1 == 0:
        print('Episode: %d, loss: %.3f, mean_reward: %.3f' % (i_episode, loss.item(), float(mean_reward)))

    # 停时条件
    if mean_reward >= 18:
        print("Solved!")
        torch.save(net.state_dict(), identity + '.pth')
        break
