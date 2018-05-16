import os

import torch as T
import torchvision as tv
from torch.autograd import Variable

from mnist_cl import mnist_cl, criterion, test_iter, DEVICE
from utils import get_cls_accuracy

model = mnist_cl()
model.load_state_dict(T.load('mnist_cl.pt'))
model.train()
model.to(DEVICE)

save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)

for i, (im, l) in enumerate(test_iter):
    x = Variable(T.cuda.FloatTensor(im.numpy()), requires_grad=True)

    l = l.to(DEVICE)
    x.grad = None
    logits = model(x)
    loss = criterion(logits, l)
    loss.backward()

    adv_noise = 0.2 * T.sign(x.grad.data)
    adv_x = T.clamp(x.data + adv_noise, 0, 1)

    tv.utils.save_image(x, save_dir + '%d.png' % i)
    tv.utils.save_image(adv_x, save_dir + '%d_adv.png' % i)

    with T.no_grad():
        pred2 = model(x)
        acc2 = get_cls_accuracy(pred2, l)
        pred1 = model(adv_x)
        acc1 = get_cls_accuracy(pred1, l)
        print('acc1: %.3f acc2: %.3f' % (acc1, acc2))
