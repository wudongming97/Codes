import os
import torch
from torch.autograd import Variable
import vanlia_vae_pt as vanlia_vae
from my_plot import my_plot


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


# 每个 epoch 查看下图片效果
def test_epoch(e):
    z = Variable(torch.randn(vanlia_vae.batch_sz, vanlia_vae.dim_z))
    samples = vanlia_vae.model.decoder(z).cpu().data.numpy()
    my_plot(vanlia_vae.save_dir, e, samples, 36)


# 主函数
if __name__ == '__main__':
    saved_model = 'vanlia_vae_pt.pkl'
    if os.path.isfile(saved_model):
        vanlia_vae.model.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))
    else:
        # train
        for e in range(50):
            vanlia_vae.train(e)
            test_epoch(e)
        torch.save(vanlia_vae.model.state_dict(), saved_model)

    test_1()
    test_2()
    # test_3()
    # test_4()
