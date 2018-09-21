# ref https://github.com/henripal/sgld/blob/master/sgld/sgld/sgld_optimizer.py
import numpy as np
import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required

from utils import DEVICE


class SGLD(Optimizer):
    def __init__(self, params, lr=0.001, addnoise=True):
        defaults = dict(lr=lr, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr=None):
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        2 * torch.ones(size) * np.sqrt(group['lr'])
                    )
                    p.data.add_(-group['lr'], d_p + langevin_noise.sample().to(DEVICE))
                else:
                    p.data.add_(-group['lr'], d_p)
        return None


class pSGLD(Optimizer):
    def __init__(self, params, lr=required, alpha=0.99, eps=1e-5, centered=False, addnoise=True):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)

    def step(self, lr=None):
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size).cuda(),
                        torch.ones(size).cuda().mul_(group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'], d_p.div_(avg) + langevin_noise.sample())
                else:
                    p.data.addcdiv_(-group['lr'], d_p, avg)

        return None
