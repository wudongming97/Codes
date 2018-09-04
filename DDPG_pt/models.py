import torch
import torch.nn as nn

from utils import DEVICE


class RDPG_Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(RDPG_Actor, self).__init__()
        self.fc_in = nn.Linear(obs_size, 128)
        self.gru = nn.GRUCell(128, 128)
        self.fc_out = nn.Linear(128, act_size)
        # weight init
        self.fc_in.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.to(DEVICE)

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc_in(x))
        hx = self.gru(x, hidden)
        x = torch.tanh(self.fc_out(hx))
        return x, hx

    def get_action(self, state, hidden=None):
        state = torch.FloatTensor(state).to(DEVICE)
        action, hx = self.forward(state, hidden)
        return action.detach().cpu().numpy(), hx


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

    def get_action(self, state):
        state = torch.FloatTensor(state).to(DEVICE)
        action = self.forward(state)
        return action.detach().cpu().numpy()


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
