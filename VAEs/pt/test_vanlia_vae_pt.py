import os

import torch
from torch.autograd import Variable
from utils import my_plot

from pt import vanlia_vae_pt as vanlia_vae


# 获取（num）张制定（digit）的图片
def get_special_imags(digit, num):
    '''

    :param digit:
    :param num:
    :return: 返回一个 list，元素为torch.FloatTensor
    '''

    imgs = torch.Tensor(num, 28*28)
    i = 0
    for data in enumerate(vanlia_vae.train_datasets):
        _, (img, label) = data
        if label == digit:
            imgs[i, :] = img.view(28*28)
            i = i + 1
        if i == num:
            break
    return imgs


# 测试用制定数字图片生成的隐变量来生成
def test_1():
    imags = get_special_imags(8, 36)
    imags = Variable(imags)
    res, _, _ = vanlia_vae.model.forward(imags)
    my_plot(vanlia_vae.save_dir, 'test_1', res.data.numpy(), 36)


# 查看用数字1和数字8的隐变量之间的连续插值生成图片
def test_2():
    imags_8 = get_special_imags(8, 36)
    imags_1 = get_special_imags(1, 36)
    delta = (imags_8 - imags_1)/20
    for i in range(20):
        res, _, _ = vanlia_vae.model.forward(Variable(imags_1 + delta * i))
        my_plot(vanlia_vae.save_dir, 'test_2_{}'.format(i), res.data.numpy(), 36)


# 测试隐变量的期望和方差
def test_3():
    None


# 可视化隐变量
def test_4():
    # todo
    None


# vae 跟 infogan 有很多类似的地方，因此猜测 vae 也能像 infogan 一样学习到 disentangle representation
def test_5():
    imags_8 = get_special_imags(8, 1)
    res, _, _ = vanlia_vae.model.forward(imags_8)
    z_8 = res.expand(36, -1)



# 用全0和全1作为输入， 会得到什么结果
def test_6():
    z0 = Variable(torch.zeros(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z0).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_6_0", samples, 36)

    z0_5 = Variable(torch.zeros(vanlia_vae.batch_sz, vanlia_vae.dim_z)*0.5)
    samples = vanlia_vae.model.decoder(z0_5).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_6_0_5", samples, 36)

    z1 = Variable(torch.ones(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z1).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_6_1", samples, 36)


# 用均值在0.5，方差为1的高斯分布作为输入
def test_7():
    z = Variable(0.5 + torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_7", samples, 64)


# 用均值在1，方差为1的高斯分布作为输入
def test_8():
    z = Variable(1 + torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_8", samples, 64)


# 用均值在2，方差为1的高斯分布作为输入
def test_9():
    z = Variable(2 + torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_9", samples, 64)

# 用均值在3，方差为1的高斯分布作为输入
def test_10():
    z = Variable(3 + torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_10", samples, 64)

# 用均值在8，方差为1的高斯分布作为输入
def test_11():
    z = Variable(8 + torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_11", samples, 64)

# 每个 epoch 查看下图片效果
def test_0():
    z = Variable(torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, "test_0", samples, 36)


# 主函数
if __name__ == '__main__':
    saved_model = 'vanlia_vae_pt.pkl'
    if os.path.isfile(saved_model):
        vanlia_vae.model.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))
    else:
        # train
        for e in range(50):
            vanlia_vae.train(e)
        torch.save(vanlia_vae.model.state_dict(), saved_model)

    # test_0()
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_5()
    # test_6()
    # test_7()
    # test_8()
    # test_9()
    # test_10()
    test_11()
