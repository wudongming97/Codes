import gym
import torch

from env_wrappers import ActionNormalizedEnv
from models import RDPG_Actor

model_name = 'rdpg_01'
env_id = "Pendulum-v0"
identity = model_name + '_' + env_id
env = ActionNormalizedEnv(gym.make(env_id))

obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]
act_net = RDPG_Actor(obs_size, act_size)
act_net.load_state_dict(torch.load(identity + '_act.pth'))


def test_policy(actor, env, vis=False, n_episodes=2, max_len=500):
    returns = []
    for i_episode in range(n_episodes):
        state = env.reset()
        if vis: env.render()
        episode_return = 0
        hx = None
        for t in range(max_len):
            action, hx = actor.get_action([state], hx)
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if vis: env.render()
            if done:
                returns.append(episode_return)
                break
    return sum(returns) / len(returns)


mean_return = test_policy(act_net, env, True)
print('mean_return: %.3f' % mean_return)
