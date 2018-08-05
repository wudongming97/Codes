import gym

N_EPISODES = 2

from models import *
from utils import *

model_name = 'PG_A2C_01'
# model_name = 'ReinforceNaive_01'
env_id = "CartPole-v0"
identity = env_id + '_' + model_name
env = gym.make(env_id)
net = NaiveAC(env.observation_space.shape[0], env.action_space.n)
net.load_state_dict(torch.load(identity + '.pth'))

for i_episode in range(N_EPISODES):
    total_reward = test_policy(net, env, True)
    print('Episode: %d total_reward: %.3f' % (i_episode, total_reward))
