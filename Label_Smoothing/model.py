import torch.nn as nn

from utils import DEVICE


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.fill_(0.0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 10)
        )
        self.to(DEVICE)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)
