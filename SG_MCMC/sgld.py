import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required

from utils import DEVICE


class SGLD(Optimizer):
    def __init__(self, params, lr=required, addnoise=True):
        defaults = dict(lr=lr, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) * np.sqrt(group['lr'])
                    )
                    p.data.add_(-group['lr'], d_p + langevin_noise.sample().to(DEVICE))
                else:
                    p.data.add_(-group['lr'], d_p)
        return None
