import gym

N_EPISODES = 2

from models import *
from utils import *

model_name = 'PG_NaiveAC'
# model_name = 'ReinforceNaive_01'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name
env = gym.make(env_id)
net = NaiveAC(env.observation_space.shape[0], env.action_space.n)
net.load_state_dict(torch.load(identity + '.pth'))

for i_episode in range(N_EPISODES):
    state = env.reset()
    env.render()
    total_reward = 0
    is_done = False
    while not is_done:
        action, _, _, _ = net.action_and_logprob(state)
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
        env.render()
    print('Episode: %d total_reward: %.3f' % (i_episode, total_reward))
