import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def action(self, x):
        logits = self.forward(x).squeeze()
        probs = F.softmax(logits, -1)
        return np.random.choice(self.n_actions, p=probs.cpu().data.numpy())
