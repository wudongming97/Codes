import torch
import torch.nn as nn

from utils import DEVICE


class DDPG_Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPG_Actor, self).__init__()
        _block = [
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        ]
        _block[-2].weight.data.uniform_(-3e-3, 3e-3)

        self.actor = nn.Sequential(*_block)
        self.to(DEVICE)

    def forward(self, x):
        return self.actor(x)

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]


class DDPG_Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPG_Critic, self).__init__()
        _block = [
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        ]
        _block[-1].weight.data.uniform_(-3e-3, 3e-3)

        self.critic = nn.Sequential(*_block)
        self.to(DEVICE)

    def forward(self, x, a):
        return self.critic(torch.cat([x, a], -1))
