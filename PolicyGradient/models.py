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
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        m = Categorical(F.softmax(self.forward(state)))
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
