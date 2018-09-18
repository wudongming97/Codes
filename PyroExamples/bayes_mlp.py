import math

import pyro
import pyro.distributions as dist_pr
import pyro.optim as optim_pr
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO

from utils import *

train_iter, test_iter = mnist_loaders("../../Datasets/MNIST/", 100)

i_features, h_features, o_features = 784, 1200, 10


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(i_features, h_features)
        self.l2 = nn.Linear(h_features, o_features)
        self.to(DEVICE)

    def forward(self, x):
        x = x.view(-1, i_features)
        x = torch.relu(self.l1(x))
        return self.l2(x)

    def acc(self, valid_iter):
        acc = 0
        for x, y in valid_iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            acc += get_cls_accuracy(self(x), y).item()
        return acc / len(valid_iter)


mlp = MLP()


def _normal_prior(*shape):
    loc = torch.zeros(*shape).to(DEVICE)
    scale = math.exp(-3) * torch.ones(*shape).to(DEVICE)
    return dist_pr.Normal(loc, scale)


def model(x, y):
    data_len = len(train_iter) * x.size(0)
    theta = {'l1.weight': _normal_prior(h_features, i_features),
             'l1.bias': _normal_prior(h_features),
             'l2.weight': _normal_prior(o_features, h_features),
             'l2.bias': _normal_prior(o_features)}
    lifted_mlp = pyro.random_module('bayes_mlp', mlp, theta)()
    with pyro.iarange('data', size=data_len, subsample_size=x.size(0)):
        logits = lifted_mlp(x)
        pyro.sample('obs', dist_pr.Categorical(logits=logits), obs=y)


def guide(x, y):
    l1_w_loc = pyro.param('l1_w_loc', torch.zeros(h_features, i_features))
    l1_w_scale = 0.01 * pyro.param('l1_w_scale', torch.randn(h_features, i_features)).exp()
    l1_b_loc = pyro.param('l1_b_loc', torch.zeros(h_features))
    l1_b_scale = pyro.param('l1_b_scale', torch.randn(h_features)).exp()
    l2_w_loc = pyro.param('l2_w_loc', torch.zeros(o_features, h_features))
    l2_w_scale = 0.01 * pyro.param('l2_w_scale', torch.randn(o_features, h_features)).exp()
    l2_b_loc = pyro.param('l2_b_loc', torch.zeros(o_features))
    l2_b_scale = pyro.param('l2_b_scale', torch.randn(o_features)).exp()

    qtheta = {'l1.weight': dist_pr.Normal(l1_w_loc.to(DEVICE), l1_w_scale.to(DEVICE)),
              'l1.bias': dist_pr.Normal(l1_b_loc.to(DEVICE), l1_b_scale.to(DEVICE)),
              'l2.weight': dist_pr.Normal(l2_w_loc.to(DEVICE), l2_w_scale.to(DEVICE)),
              'l2.bias': dist_pr.Normal(l2_b_loc.to(DEVICE), l2_b_scale.to(DEVICE))}
    return pyro.random_module('bayes_mlp', mlp, qtheta)()


def train(n_epochs):
    trainer = optim_pr.Adam({"lr": 0.001})
    svi = SVI(model, guide, trainer, loss=Trace_ELBO())
    pyro.clear_param_store()

    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(train_iter):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            loss = svi.step(x, y)

            if i % 100 == 0:
                val_net = guide(None, None)
                acc = val_net.acc(test_iter)
                print("[Epoch: %d] [Batch: %d] [Loss: %d] [Acc: %.3f]" % (epoch, i, loss, acc))


if __name__ == '__main__':
    train(20)
