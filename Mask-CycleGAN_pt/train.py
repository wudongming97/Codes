import os
import sys

import torch as T
import torch.optim as optim
import torchvision as tv
from torch.autograd import Variable
from torch.optim import lr_scheduler

from networks import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# T.cuda.set_device(0)

# data
data_source = 'celeba_smile'  # celeba_smile, apple2orange,horse2zebra
save_dir = './results_celeba_smile/'
os.makedirs(save_dir, exist_ok=True)

epoch = 0
print_every = 200
save_epoch_freq = 2

# hyparams
lr = 1e-4
batch_size = 1
n_epochs = 100
lambda_gan = 0.8
lambda_cycle = 0.8
lambda_identity = 2

# network
input_nc = 3
output_nc = 4
ngf = 64
ndf = 64

patch_d = False


def get_D(patch_d_):
    if patch_d_:
        return NLayerDiscriminator(input_nc, ndf).cuda()
    else:
        return TowerDiscriminator().cuda()


netG_A = ResnetGenerator(input_nc, output_nc, ngf).cuda()
netG_B = ResnetGenerator(input_nc, output_nc, ngf).cuda()
netD_A = get_D(patch_d)
netD_B = get_D(patch_d)

# optim
opt_G = optim.Adam(list(netG_A.parameters()) + list(netG_B.parameters()), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(list(netD_A.parameters()) + list(netD_B.parameters()), lr=lr, betas=(0.5, 0.999))
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.5)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=10, gamma=0.5)

criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_gan = nn.MSELoss()


def zero_grad():
    netG_A.zero_grad()
    netG_B.zero_grad()
    netD_A.zero_grad()
    netD_B.zero_grad()


# train
print('Training...')

netG_A.train()
netG_B.train()
netD_B.train()
netD_A.train()


# loader
def get_data_loader(ds):
    if ds == 'horse2zebra':
        from coco_sub import horse2zebra_loader
        return horse2zebra_loader
    elif ds == 'apple2orange':
        from coco_sub import apple2orange_loader
        return apple2orange_loader
    elif ds == 'celeba_smile':
        from celeba import celeba_smile_loader
        return celeba_smile_loader
    else:
        raise ValueError


data_loader = get_data_loader(data_source)

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
    for A, B in data_loader:
        batch += 1
        # G:A -> B
        a_real = Variable(A).cuda()
        b_fake, b_oimg, b_mask = netG_A(a_real)
        b_fake_score = netD_B(b_fake)
        a_rec, _, _ = netG_B(b_fake)

        loss_A2B_gan = criterion_gan(b_fake_score, T.ones_like(b_fake_score) * 0.9)
        loss_A2B_cyc = criterion_cycle(a_rec, a_real)
        loss_A2B_idt = criterion_identity(b_fake, a_real)

        # F:B->A
        b_real = Variable(B).cuda()
        a_fake, a_oimg, a_mask = netG_B(b_real)
        a_fake_score = netD_A(a_fake)
        b_rec, _, _ = netG_A(a_fake)

        loss_B2A_gan = criterion_gan(a_fake_score, T.ones_like(a_fake_score) * 0.9)
        loss_B2A_cyc = criterion_cycle(b_rec, b_real)
        loss_B2A_idt = criterion_identity(a_fake, b_real)

        loss_G = ((loss_A2B_gan + loss_B2A_gan) * lambda_gan +
                  (loss_A2B_cyc + loss_B2A_cyc) * lambda_cycle +
                  (loss_A2B_idt + loss_B2A_idt) * lambda_identity)

        zero_grad()
        loss_G.backward()
        opt_G.step()

        # train D
        b_fake_score1 = netD_B(b_fake.detach())
        b_real_score1 = netD_B(b_real.detach())
        loss_D_b = (criterion_gan(b_fake_score1, T.ones_like(b_fake_score1) * 0.1)
                    + criterion_gan(b_real_score1, T.ones_like(b_real_score1) * 0.9))
        a_fake_score1 = netD_A(a_fake.detach())
        a_real_score1 = netD_A(a_real.detach())
        loss_D_a = (criterion_gan(a_fake_score1, T.ones_like(a_fake_score1) * 0.1)
                    + criterion_gan(a_real_score1, T.ones_like(a_real_score1) * 0.9))
        loss_D = loss_D_a + loss_D_b

        zero_grad()
        loss_D.backward()
        opt_D.step()

        if batch % print_every == 0:
            print('Epoch #%d' % epoch)
            print('Batch #%d' % batch)

            print('Loss D: %0.3f' % loss_D.data[0] + '\t' +
                  'Loss G: %0.3f' % loss_G.data[0])
            print('Loss P2N G real: %0.3f' % loss_A2B_gan.data[0] + '\t' +
                  'Loss N2P G fake: %0.3f' % loss_B2A_gan.data[0])

            print('-' * 50)
            sys.stdout.flush()

            tv.utils.save_image(T.cat([
                a_real.data * 0.5 + 0.5,
                b_fake.data * 0.5 + 0.5,
                b_oimg.data * 0.5 + 0.5,
                b_mask.data,
                a_rec.data * 0.5 + 0.5,
                # --------------------
                b_real.data * 0.5 + 0.5,
                a_fake.data * 0.5 + 0.5,
                a_oimg.data * 0.5 + 0.5,
                a_mask.data,
                b_rec.data * 0.5 + 0.5], 0),
                save_dir + 'train_{}_{}.png'.format(epoch, batch), 5)

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
