# vae可以看做在包含隐变量 z 模型的生成器模型的基础上，再增加了一个正则项，迫使隐变量 z 满足某种分布。
# vanlia_generative_model 主要想尝试下不要这个正则项，生成会有什么效果。
# 模型描述：z~N(0，1)为隐变量，先验概率为高斯分布，通过一个由神经网络实现的确定性函数 f 把 z 变换到数据空间，通过最大似然方法训练模型。
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import MNIST_train_loader, my_plot, Batch_sz

batch_sz = Batch_sz
saved_model = 'vanlia_generative_model.pkl'
save_dir = 'out/vanlia_generative_model/'

class vanlia_generative_model(nn.Module):
    def __init__(self, dataset = 'mnist'):
        super(vanlia_generative_model, self).__init__()
        if dataset == 'mnist':
            self.dim_z = 32
            self.dim_img = 28*28
        self.generator = nn.Sequential(
            torch.nn.Linear(self.dim_z, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.dim_img),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.dim_img, self.dim_img)
        )
    def forward(self, z):
        return self.generator(z)


model = vanlia_generative_model()
optimizer = torch.optim.Adam(model.parameters())
mse_loss = nn.MSELoss()


def train_ep(epoch):

    for batch_idx, (data, _) in enumerate(MNIST_train_loader):
        z = Variable(torch.randn(batch_sz, model.dim_z))
        imgs = Variable(data)

        optimizer.zero_grad()
        loss = mse_loss(model(z), imgs)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch: [%2d] Batch_idx:[%4d]  loss: %.8f" %
                  ((epoch), (batch_idx), loss.data[0]))


def train():
    for e in range(50):
        train_ep(e)
    torch.save(model.state_dict(), saved_model)


def test_0():
    z = Variable(torch.randn(batch_sz, model.dim_z))
    samples = model(z).data.numpy()
    my_plot(save_dir, "test_0", samples, 36)


if __name__ == '__main__':
    if os.path.isfile(saved_model):
        model.load_state_dict(torch.load(saved_model))
    else:
        train()
    test_0()
    #test_1()