# pytorch 0.4
import os
import sys

import torch as T
import torch.optim as optim
import torchvision as tv
from torch.optim import lr_scheduler

from celeba import celeba_gender_loader, batch_size
from networks import *

use_cuda = T.cuda.is_available()
device = T.device('cuda' if use_cuda else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# T.cuda.set_device(0)

# data
save_dir = './results_celeba_gender/'
os.makedirs(save_dir, exist_ok=True)

epoch = 0
print_every = 200
save_epoch_freq = 1

# hyparams
lr = 1e-4
n_epochs = 100
lambda_gan = 0.8
lambda_cycle = 5
lambda_cls = 0.2
lambda_identity = 0.00

# network
input_nc = 3
output_nc = 3
ngf = 32
ndf = 32

netG_A = ResnetGenerator(input_nc, output_nc, ngf, nn.BatchNorm2d).to(device)
netG_B = ResnetGenerator(input_nc, output_nc, ngf, nn.BatchNorm2d).to(device)
netD_A = NLayerDiscriminator(input_nc, ndf, nn.BatchNorm2d).to(device)
netD_B = NLayerDiscriminator(input_nc, ndf, nn.BatchNorm2d).to(device)

# optim
opt_G = optim.Adam(list(netG_A.parameters()) + list(netG_B.parameters()), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(list(netD_A.parameters()) + list(netD_B.parameters()), lr=lr, betas=(0.5, 0.999))
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.5)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=10, gamma=0.5)

criterion_cycle = nn.L1Loss()
criterion_gan = nn.MSELoss()
criterion_cls = nn.CrossEntropyLoss()
criterion_identity = nn.L1Loss()


def zero_grad():
    netG_A.zero_grad()
    netG_B.zero_grad()
    netD_A.zero_grad()
    netD_B.zero_grad()


def get_cls_accuracy(score, label):
    total = label.size()[0]
    _, pred = T.max(score, dim=1)
    correct = T.sum(pred == label)
    accuracy = correct / total

    return accuracy


# train
print('Training...')

netG_A.train()
netG_B.train()
netD_A.train()
netD_B.train()

# resume
if epoch >= 1:
    checkpoint = T.load(save_dir + 'ckpt_{}.ptz'.format(epoch))
    lr = checkpoint['lr']
    epoch = checkpoint['epoch']
    netG_A.load_state_dict(checkpoint['G_A'])
    netG_B.load_state_dict(checkpoint['G_B'])
    netD_A.load_state_dict(checkpoint['D_A'])
    netD_B.load_state_dict(checkpoint['D_B'])

for _ in range(epoch, n_epochs):
    epoch += 1
    if epoch > 50:
        scheduler_G.step()
        scheduler_D.step()

    batch = 0
    for (A, lA), (B, lB) in celeba_gender_loader:
        batch += 1
        lA = lA.to(device)
        lB = lB.to(device)
        # G:A -> B
        a_real = A.to(device)
        b_fake = netG_A(a_real)
        b_fake_score, b_fake_cls = netD_B(b_fake)
        b_fake_acc = get_cls_accuracy(b_fake_cls, lB)
        a_rec = netG_B(b_fake)

        loss_A2B_gan = criterion_gan(b_fake_score, T.ones_like(b_fake_score) * 0.9)
        loss_A2B_cyc = criterion_cycle(a_rec, a_real)
        loss_A2B_cls = criterion_cls(b_fake_cls, lB)
        loss_A2B_idt = criterion_identity(b_fake, a_real)

        # F:B->A
        b_real = B.to(device)
        a_fake = netG_B(b_real)
        a_fake_score, a_fake_cls = netD_A(a_fake)
        a_fake_acc = get_cls_accuracy(a_fake_cls, lA)
        b_rec = netG_A(a_fake)

        loss_B2A_gan = criterion_gan(a_fake_score, T.ones_like(a_fake_score) * 0.9)
        loss_B2A_cyc = criterion_cycle(b_rec, b_real)
        loss_B2A_cls = criterion_cls(a_fake_cls, lA)
        loss_B2A_idt = criterion_identity(a_fake, b_real)

        loss_G = ((loss_A2B_gan + loss_B2A_gan) * lambda_gan +
                  (loss_A2B_cls + loss_B2A_cls) * lambda_cls +
                  (loss_A2B_idt + loss_B2A_idt) * lambda_identity +
                  (loss_A2B_cyc + loss_B2A_cyc) * lambda_cycle)

        zero_grad()
        loss_G.backward()
        opt_G.step()

        # train D
        b_fake_score1, _ = netD_B(b_fake.detach())
        b_real_score1, b_real_cls = netD_B(b_real)
        b_real_acc = get_cls_accuracy(b_real_cls, lB)
        loss_D_b = (criterion_gan(b_fake_score1, T.ones_like(b_fake_score1) * 0.1) +
                    criterion_gan(b_real_score1, T.ones_like(b_real_score1) * 0.9))
        a_fake_score1, _ = netD_A(a_fake.detach())
        a_real_score1, a_real_cls = netD_A(a_real)
        a_real_acc = get_cls_accuracy(a_real_cls, lA)
        loss_D_a = (criterion_gan(a_fake_score1, T.ones_like(a_fake_score1) * 0.1)
                    + criterion_gan(a_real_score1, T.ones_like(a_real_score1) * 0.9))
        loss_D_cls = criterion_cls(b_real_cls, lB) + criterion_cls(a_real_cls, lA)
        loss_D = loss_D_a + loss_D_b + loss_D_cls

        zero_grad()
        loss_D.backward()
        opt_D.step()

        if batch % print_every == 0:
            print('Epoch #%d' % epoch)
            print('Batch #%d' % batch)

            print('Loss D: %0.3f' % loss_D.item() + '\t' +
                  'Loss G: %0.3f' % loss_G.item())
            print('Loss P2N G real: %0.3f' % loss_A2B_gan.item() + '\t' +
                  'Loss N2P G fake: %0.3f' % loss_B2A_gan.item())
            print('Acc a_fake: %0.3f' % a_fake_acc.item() + '\t' +
                  'Acc b_fake: %0.3f' % b_fake_acc.item() + '\t' +
                  'Acc a_real: %0.3f' % a_real_acc.item() + '\t' +
                  'Acc b_real: %0.3f' % b_real_acc.item())

            print('-' * 50)
            sys.stdout.flush()

            tv.utils.save_image(T.cat([
                a_real.data * 0.5 + 0.5,
                b_fake.data * 0.5 + 0.5,
                a_rec.data * 0.5 + 0.5,
                # --------------------
                b_real.data * 0.5 + 0.5,
                a_fake.data * 0.5 + 0.5,
                b_rec.data * 0.5 + 0.5], 0),
                save_dir + 'train_{}_{}.png'.format(epoch, batch), batch_size)

    if epoch % save_epoch_freq == 0:
        T.save({
            'epoch': epoch,
            'lr': lr,
            'G_A': netG_A.state_dict(),
            'G_B': netG_B.state_dict(),
            'D_A': netD_A.state_dict(),
            'D_B': netD_B.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict()
        }, save_dir + 'ckpt_{}.ptz'.format(epoch))
