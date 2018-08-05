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

    def action_dist(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        return Categorical(F.softmax(self.forward(state), -1))


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

    def forward(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        action_logit = self.fc(state)
        return Categorical(F.softmax(action_logit, -1))


class NaiveAC(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(NaiveAC, self).__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.common = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU()
        )
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        self.to(DEVICE)

    def forward(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        out = self.common(state)
        action_logit, value = self.action_head(out), self.value_head(out)
        return Categorical(F.softmax(action_logit, -1)), value


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()
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

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.to(DEVICE)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        out = self.conv(state).view(state.size(0), -1)
        return Categorical(F.softmax(self.policy(out), -1)), self.value(out)
