# coding: utf-8
# the code of class generator and discriminator is copied from https://github.com/znxlwm/pytorch-generative-model-collections.git

# 本实验打算实现一个最基本的infogan，同时探讨以下几个问题：
# * 验证 infogan 能学习到的 disentangled 表达。
# * 改变 disentangled factor 的个数，查看学到的 factor 有什么不同。
# * 共享和不共享 D 的不同。
# * 探讨为什么这种方法可以达到这种目的。

# lib
import os
import itertools
import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import MNIST_train_loader, Batch_sz

# var
save_dir = 'out/infogan/'
use_gpu = torch.cuda.is_available()

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.z_dim = 62
            self.input_dim = self.z_dim + 12
            self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
            self.len_discrete_code = 10  # categorical distribution (i.e. label)
            self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        a = F.sigmoid(x[:, self.output_dim])
        cont_code = x[:, self.output_dim:self.output_dim + self.len_continuous_code]
        dist_code = x[:, self.output_dim + self.len_continuous_code:]

        return a, cont_code, dist_code


# infogan
class InfoGAN(nn.Module):
    def __init__(self, share=True):
        super(InfoGAN, self).__init__()

        # hyper paramters
        self.batch_size = Batch_sz
        self.lr = 0.001

        # var
        self.save_dir = './model/'
        self.model_name = 'infogan'

        self.train_dataloader = MNIST_train_loader
        self.G = generator()
        self.D = discriminator()
        self.G_optimizer = torch.optim.Adam(self.G.parameters())
        self.D_optimizer = torch.optim.Adam(self.D.parameters())
        self.info_optimizer = torch.optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()))
        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()

        self.y_real = Variable(torch.ones(self.batch_size, 1))
        self.y_fake = Variable(torch.zeros(self.batch_size, 1))


    def train_ep(self, epoch):
        train_loss = []
        total_batch = 0
        for batch_idx, (data, _) in enumerate(self.train_dataloader):
            input = Variable(data)
            z_code = Variable(torch.randn(self.batch_size, self.G.z_dim))
            cont_code = Variable(torch.rand(self.batch_size, 2))  # 0-1
            dist_label = torch.LongTensor(self.batch_size, 1).random_() % self.D.len_discrete_code
            dist_code = Variable(torch.zeros(self.batch_size, self.D.len_discrete_code).scatter_(1, dist_label, 1))

            # update d
            self.D_optimizer.zero_grad()
            d_real, _, _ = self.D(input)
            d_fake, _, _ = self.D(self.G(z_code, cont_code, dist_code))
            d_loss = self.BCE_loss(d_real, self.y_real) + self.BCE_loss(d_fake, self.y_fake)
            d_loss.backward(retain_graph=True)
            self.D_optimizer.step()

            # update g
            self.G_optimizer.zero_grad()
            d_fake, d_cont, d_disc = self.D(self.G(z_code, cont_code, dist_code))
            g_loss = self.BCE_loss(d_fake, self.y_real)
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()

            # update info
            disc_loss = self.CE_loss(d_disc, torch.max(dist_code, 1)[1])
            cont_loss = self.MSE_loss(d_cont, cont_code)
            info_loss = disc_loss + cont_loss
            info_loss.backward()
            self.info_optimizer.step()

            train_loss += d_loss, g_loss, info_loss
            total_batch = batch_idx

            if batch_idx % 100 == 0:
                train_loss.append((d_loss.data[0], g_loss.data[0], info_loss.data[0]))
                print("Epoch: [%2d] Batch_idx:[%4d]  D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %
                      ((epoch), (batch_idx), d_loss.data[0], g_loss.data[0], info_loss.data[0]))

        return train_loss

    def save(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, self.model_name + '_D.pkl'))
        save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl' + save_time))
        torch.save(self.D.state_dict(), os.path.join(self.save_dir, self.model_name + '_D.pkl' + save_time))

    def load(self):

        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_D.pkl')))


if __name__ == '__main__':
    model = InfoGAN()
    loss = []
    for e in range(100):
        loss.extend(model.train_ep(e))
    model.save()

    # dump loss and use it to plot
    with open('loss_pickle.pkl', 'wb') as f:
        pickle.dump(loss, f)
    # plot ？？

