from vanlia_vae_pt import *

# 获取（num）张制定（digit）的图片
def get_special_imags(digit, num):
    '''

    :param digit:
    :param num:
    :return: 返回一个 list，元素为torch.FloatTensor
    '''

    imgs = []
    i = 0
    for data in enumerate(train_datasets):
        _, (img,label) = data
        if label == digit:
            imgs.append(img)
            print(label)
            i = i + 1
        if i == num:
            break
    return imgs


# 测试用制定数字图片生成的隐变量来生成
def test_1():
    # todo
    None


# 查看用数字1和数字8的隐变量之间的连续插值生成图片
def test_2():
    # todo
    None

# 测试隐变量的期望和方差
def test_3():
    # todo
    None

# 可视化隐变量
def test_4():
    # todo
    None

# 每个 epoch 查看下图片效果
def test_epoch(e):
    z = Variable(torch.randn(batch_sz, dim_z))
    samples = model.decoder(z).cpu().data.numpy()
    my_plot(save_dir, e, samples, 36)

# 主函数
if __name__ == '__main__':
    saved_model = 'vanlia_vae_pt.pkl'
    if os.path.isfile(saved_model):
        model.load_state_dict(torch.load(saved_model))
    else:
        # train
        for e in range(100):
            train(e)
            test_epoch(e)
        torch.save(model.state_dict(), saved_model)

    test_1()


