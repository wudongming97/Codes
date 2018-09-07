import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


def guassian(x, mu, sigma):
    scaling = 1.0 / torch.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = torch.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
    return scaling * bell


def log_gaussian(x, mu, sigma):
    return -0.5 * math.log(2.0 * np.pi) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)


def gaussian_prior(x, sigma=math.exp(-3)):
    sigma = torch.tensor(sigma).to(DEVICE)
    return log_gaussian(x, 0., sigma)


def scale_mixture_prior(x, sigma_p1=math.exp(-1), sigma_p2=math.exp(-6), pi=0.5):
    sigma_p1 = torch.tensor(sigma_p1).to(DEVICE)
    sigma_p2 = torch.tensor(sigma_p2).to(DEVICE)
    g1 = pi * guassian(x, 0., sigma_p1)
    g2 = (1 - pi) * guassian(x, 0., sigma_p2)
    return torch.log(g1 + g2)


class BBLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_type='mixture'):
        super(BBLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_mu = nn.Parameter(torch.Tensor(in_features, out_features))
        self.W_rho = nn.Parameter(torch.Tensor(in_features, out_features))
        self.B_mu = nn.Parameter(torch.Tensor(out_features))
        self.B_rho = nn.Parameter(torch.Tensor(out_features))
        if prior_type == 'mixture':
            self.prior = scale_mixture_prior
        else:
            self.prior = gaussian_prior
        self.kl_loss = 0
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化很重要 ref:https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon.html
        nn.init.normal_(self.W_mu, std=0.1)
        nn.init.constant_(self.W_rho, -3)
        nn.init.normal_(self.B_mu, std=0.1)
        nn.init.constant_(self.B_rho, -3)

    def forward(self, x, infer=False):
        if infer:
            return x @ self.W_mu + self.B_mu
        else:
            epsilon_w = torch.randn(self.in_features, self.out_features).to(DEVICE)
            epsilon_b = torch.randn(self.out_features).to(DEVICE)
            sigma_w = F.softplus(self.W_rho)
            sigma_b = F.softplus(self.B_rho)
            w = self.W_mu + sigma_w * epsilon_w
            b = self.B_mu + sigma_b * epsilon_b

            # 计算kl_loss
            posterior_w = log_gaussian(w, self.W_mu, sigma_w).sum()
            posterior_b = log_gaussian(b, self.B_mu, sigma_b).sum()
            prior_w = self.prior(w).sum()
            prior_b = self.prior(b).sum()
            self.kl_loss = posterior_w + posterior_b - prior_w - prior_b

            return x @ w + b
