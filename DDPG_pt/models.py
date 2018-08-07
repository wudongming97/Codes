import torch
import torch.nn as nn

from utils import DEVICE


class DDPG_Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPG_Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )
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
        self.h1 = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.to(DEVICE)

    def forward(self, x, a):
        h1 = self.h1(x)
        return self.out(torch.cat([h1, a], -1))
