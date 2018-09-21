import matplotlib.pyplot as plt
import torch.nn as nn

from sgld import *
from utils import *


class TiedMixtureGuass(nn.Module):
    def __init__(self, sigma1, sigma2, sigmax):
        super(TiedMixtureGuass, self).__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigmax = sigmax
        self.theta_1 = nn.Parameter(torch.randn(1))
        self.theta_2 = nn.Parameter(torch.randn(1))
        self.to(DEVICE)

    def forward(self, x):
        log_theta1_prior = log_gaussian(self.theta_1, 0, self.sigma1)
        log_theta2_prior = log_gaussian(self.theta_2, 0, self.sigma2)
        log_px = -2 * math.log(2) + log_gaussian(x, self.theta_1, self.sigmax) + \
                 log_gaussian(x, self.theta_1 + self.theta_2, self.sigmax)
        return log_theta1_prior + log_theta2_prior, log_px.sum()


def synthetic_data(theta1, theta2, sigmax, num=100):
    """
    从混合高斯中采样，与上面的MixtureGuass有相同的生成过程
    """
    flag = torch.randint(2, [num])
    return flag * torch.normal(theta1 * torch.ones(num), sigmax) + (1 - flag) * torch.normal(
        (theta1 + theta2) * torch.ones(num), sigmax)


if __name__ == '__main__':
    theta1 = 0
    theta2 = 1
    sigma1 = math.sqrt(10)
    sigma2 = 1
    sigmax = math.sqrt(2)

    init_lr = 0.01
    last_lr = 0.0001
    batch_size = 1
    total_iter_num = 1000000

    model = TiedMixtureGuass(sigma1, sigma2, sigmax)
    trainer = pSGLD(model.parameters(), init_lr)

    data_iter = tensor_loader(
        data=synthetic_data(theta1, theta2, sigmax, num=100),
        batch_size=batch_size
    )
    n_batchs = len(data_iter)
    # train
    samples_theta1 = []
    samples_theta2 = []
    updated_lr = init_lr
    for t in range(total_iter_num):
        x = next(iter(data_iter)).to(DEVICE)
        log_pw, log_px = model(x)
        loss = -log_pw - n_batchs * log_px
        trainer.zero_grad()
        loss.backward()
        trainer.step(updated_lr)
        updated_lr = lr_sgld_scheduler(t, init_lr, last_lr, total_steps=total_iter_num)

        # sample theta
        params = list(model.parameters())
        samples_theta1.append(params[0].item())
        samples_theta2.append(params[1].item())

        if t % 100 == 0:
            print('[T: %d] [loss: %.3f][Lr: %.8f]' % (t, loss.item(), updated_lr))

    # plot
    plt.hist2d(samples_theta1, samples_theta2, 200, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
