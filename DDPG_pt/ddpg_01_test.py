import gym
import torch

from env_wrappers import ActionNormalizedEnv
from models import DDPG_Actor
from utils import test_policy

model_name = 'ddpg_01'
env_id = "Pendulum-v0"
identity = model_name + '_' + env_id
env = ActionNormalizedEnv(gym.make(env_id))

obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]
act_net = DDPG_Actor(obs_size, act_size)
act_net.load_state_dict(torch.load(identity + '_act.pth'))

mean_return = test_policy(act_net, env, True)
print('mean_return: %.3f' % mean_return)
