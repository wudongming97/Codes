import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import *


class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.to(DEVICE)

    def forward(self, x):
        return self.net(x)

    def action_and_logprob(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
        m = Categorical(F.softmax(self.forward(state), -1))
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob


class AtariPolicyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariPolicyNet, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.to(DEVICE)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def action_and_logprob(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(DEVICE)
        m = Categorical(F.softmax(self.forward(state), -1))
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
