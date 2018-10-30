import torch
import torch.nn as nn
import torch.nn.functional as F


class STNLayer(nn.Module):
    def __init__(self, in_features):
        super(STNLayer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_features, in_features * 2, 4, 2, 1),
            nn.MaxPool2d(4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_features * 2, in_features * 2, 4, 2, 1),
            nn.MaxPool2d(4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_features * 2, in_features, 4, 2, 1),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())  # (64,256,256,2)
        x = F.grid_sample(x, grid)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, use_bias=False):
        super(Conv2dBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding, bias=use_bias),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self._block(x)


class ResnetBlock(nn.Module):
    def __init__(self, nc, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        _block = [Conv2dBlock(nc, nc, 3, 1, 1, use_bias=use_bias)]
        if use_dropout:
            _block += [nn.Dropout(0.5)]
        _block += [nn.Conv2d(nc, nc, 3, 1, 1, bias=use_bias),
                   nn.BatchNorm2d(nc)]
        self.net_block = nn.Sequential(*_block)

    def forward(self, x):
        return x + self.net_block(x)


class dsh_network(nn.Module):
    def __init__(self, input_nc, nf=32):
        super(dsh_network, self).__init__()

        self.up = nn.Sequential(
            Conv2dBlock(input_nc, nf, 4, 2, 1),
            Conv2dBlock(nf, nf * 2, 4, 2, 1),
            Conv2dBlock(nf * 2, nf * 4, 4, 2, 1),
        )
        self.down = nn.Sequential(
            Conv2dBlock(input_nc, nf, 4, 2, 1),
            Conv2dBlock(nf, nf * 2, 4, 2, 1),
            Conv2dBlock(nf * 2, nf * 4, 4, 2, 1),
        )
        self.share = nn.Sequential(
            ResnetBlock(nf * 4),
            ResnetBlock(nf * 4),
            nn.ReLU(),
            Conv2dBlock(nf * 4, nf * 2, 4, 2, 1),
            nn.Conv2d(nf * 2, 1, 4, 2, 1),  # 8 x 8
            nn.ReLU()
        )

    def forward(self, x, up=True):
        out = self.up(x) if up else self.down(x)
        out = self.share(out)
        # out = nn.parallel.data_parallel(self.share, out)
        return out.view(-1, 8 * 8)


nf = 32
input_nc = 1


def get_network():
    return dsh_network(input_nc, nf)
