import torch.nn as nn


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
