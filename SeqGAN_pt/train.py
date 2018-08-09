from itertools import count

import torch.optim as optim
from tensorboardX import SummaryWriter

from discriminator import Discriminator
from generator import Generator
from utils import *

SOS = 0
D_EPOCHS = 10
N_ROLLS = 4
EMB_SIZE = 64
HID_SIZE = 64
N_EPOCHS = 20
VOC_SIZE = 5000
BATCH_SIZE = 50
MAX_SEQ_LEN = 20
NUM_SAMPLES = 50000

# create data
oracle_samples_file = './oracle_samples.trc'
oracle_state_dict_file = './oracle.pth'
Ora = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS, True)


def _create_data():
    start = torch.zeros(NUM_SAMPLES, 1).long().to(DEVICE)
    samples = Ora.sample(start, MAX_SEQ_LEN)
    torch.save(samples.cpu(), oracle_samples_file)
    torch.save(Ora.state_dict(), oracle_state_dict_file)


_create_data()

g_data_iter = data_iter(oracle_samples_file, BATCH_SIZE)
d_data_iter = data_iter(oracle_samples_file, BATCH_SIZE)

# model
writer = SummaryWriter()
G = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS)
G_t = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS)
D = Discriminator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN)
bce_loss = nn.BCELoss()
g_trainer = optim.Adam(G.parameters())
d_trainer = optim.Adam(D.parameters())


def update_G(x):
    log_probs, outs = G.log_pt(x)
    rewards = 0
    for i in range(N_ROLLS):
        rolls = G_t.roll_out(outs)
        rewards += D.rewards(rolls)
    rewards /= N_ROLLS

    loss_pg = -(log_probs * rewards.detach()).sum(1).mean()

    g_trainer.zero_grad()
    loss_pg.backward()
    g_trainer.step()

    return loss_pg.item()


def update_D(real, fake):
    real_score = D(real)
    fake_score = D(fake)
    real_label = torch.ones_like(real_score)
    fake_label = torch.zeros_like(fake_score)
    d_loss = bce_loss(real_score, real_label) + bce_loss(fake_score, fake_label)
    d_trainer.zero_grad()
    d_loss.backward()
    d_trainer.step()
    return d_loss.item(), real_score.mean().item(), fake_score.mean().item()


def train_PG(frame_idx):
    # g
    x = next(iter(g_data_iter)).to(DEVICE)
    g_loss = update_G(x)

    # eval
    mse_loss = nll_loss(G, x)
    eval_samples = G.sample(torch.zeros(BATCH_SIZE, 1).long().to(DEVICE), MAX_SEQ_LEN)
    oracle_loss = nll_loss(Ora, eval_samples)

    # d
    d_loss, real_score, fake_score = 0, 0, 0
    for _ in range(D_EPOCHS):
        real = next(iter(d_data_iter)).to(DEVICE)
        fake = G.sample(torch.zeros(BATCH_SIZE, 1).long().to(DEVICE), MAX_SEQ_LEN)
        d_loss, real_score, fake_score = update_D(real, fake)

    print('Iter: %d, g_loss: %.3f, mse_loss: %.3f, oracle_loss: %.3f, d_loss: %.3f, r_score: %.3f, f_score: %.3f' % (
        frame_idx, g_loss, mse_loss.item(), oracle_loss.item(), d_loss, real_score, fake_score))
    return mse_loss.item(), oracle_loss.item(), g_loss, d_loss, real_score, fake_score


def pre_train(frame_idx):
    real = next(iter(g_data_iter)).to(DEVICE)
    loss = nll_loss(G, real)
    g_trainer.zero_grad()
    loss.backward()
    g_trainer.step()

    # eval
    eval_samples = G.sample(torch.zeros(BATCH_SIZE, 1).long().to(DEVICE), MAX_SEQ_LEN)
    oracle_loss = nll_loss(Ora, eval_samples)

    fake = G.sample(torch.zeros(BATCH_SIZE, 1).long().to(DEVICE), MAX_SEQ_LEN)
    d_loss, real_score, fake_score = update_D(real, fake)

    if frame_idx % 100 == 0:
        print('Iter: %d, mle_loss: %.3f, oracle_loss: %.3f, d_loss: %.3f, r_score: %.3f, f_score: %.3f' % (
            frame_idx, loss.item(), oracle_loss.item(), d_loss, real_score, fake_score))

    return loss.item(), oracle_loss.item(), d_loss, real_score, fake_score


if __name__ == '__main__':
    for frame_idx in count(1):
        if frame_idx < 30000:
            losses = pre_train(frame_idx)
            writer.add_scalars('pre_train', {'mse_loss': losses[0], 'oracle_loss': losses[1], 'd_loss': losses[2],
                                             'real_score': losses[3], 'fake_score': losses[4]}, frame_idx)
            hard_update(G_t, G)
        else:
            losses = train_PG(frame_idx)
            writer.add_scalars('train_PG', {'mse_loss': losses[0], 'oracle_loss': losses[1], 'g_loss': losses[2],
                                            'd_loss': losses[3], 'real_score': losses[4], 'fake_score': losses[5]},
                               frame_idx)
            hard_update(G_t, G)