from itertools import count

import torch.optim as optim
from tensorboardX import SummaryWriter

from discriminator import *
from generator import Generator
from utils import *

SOS = 0
D_EPOCHS = 10
N_ROLLS = 5
EMB_SIZE = 32
HID_SIZE = 32
N_EPOCHS = 20
VOC_SIZE = 5000
BATCH_SIZE = 64
MAX_SEQ_LEN = 20
NUM_SAMPLES = 10000

# create data
oracle_samples_file = './oracle_samples.trc'
oracle_state_dict_file = './oracle.pth'
Ora = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS, True)


def _create_data():
    samples = Ora.sample(NUM_SAMPLES)
    torch.save(samples.cpu(), oracle_samples_file)
    torch.save(Ora.state_dict(), oracle_state_dict_file)


_create_data()

oracle_iter = data_iter(oracle_samples_file, BATCH_SIZE)

# model
writer = SummaryWriter()
G = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS)
G_t = Generator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN, SOS)
# D = RNNDiscriminator(VOC_SIZE, EMB_SIZE, HID_SIZE, MAX_SEQ_LEN)
D = CNNDiscriminator(VOC_SIZE, EMB_SIZE)
bce_loss = nn.BCELoss()
g_trainer = optim.Adam(G.parameters(), lr=1e-2)
d_trainer = optim.Adagrad(D.parameters())


def get_d_acc(real, fake):
    real_score = D(real)
    fake_score = D(fake)
    acc = 0.5 * ((real_score > 0.5).float().mean() + (fake_score <= 0.5).float().mean())
    return acc.item()


def update_G():
    samples, log_probs = G.log_probs(BATCH_SIZE)

    rewards = 0
    for i in range(N_ROLLS):
        rolls = G_t.roll_out(samples)
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
    acc = get_d_acc(real, fake)
    return d_loss.item(), real_score.mean().item(), fake_score.mean().item(), acc


def train_PG(frame_idx):
    # g
    g_loss = update_G()

    # eval
    real = next(iter(oracle_iter)).to(DEVICE)
    mse_loss = nll_loss(G, real)
    eval_samples = G.sample(BATCH_SIZE)
    oracle_loss = nll_loss(Ora, eval_samples)

    # d
    d_loss, real_score, fake_score, acc = 0, 0, 0, 0
    for _ in range(D_EPOCHS):
        for real in oracle_iter:
            real = real.to(DEVICE)
            fake = G.sample(BATCH_SIZE)
            d_loss, real_score, fake_score, acc = update_D(real, fake)

    print(
        'Iter: %d, g_loss: %.3f, mse_loss: %.3f, oracle_loss: %.3f, d_loss: %.3f, r_score: %.3f, f_score: %.3f, acc: %.3f' % (
            frame_idx, g_loss, mse_loss.item(), oracle_loss.item(), d_loss, real_score, fake_score, acc))
    return mse_loss.item(), oracle_loss.item(), d_loss, real_score, fake_score, acc


def pre_train(frame_idx):
    real = next(iter(oracle_iter)).to(DEVICE)
    loss = nll_loss(G, real)
    g_trainer.zero_grad()
    loss.backward()
    g_trainer.step()

    # eval
    eval_samples = G.sample(BATCH_SIZE)
    oracle_loss = nll_loss(Ora, eval_samples)

    fake = G.sample(BATCH_SIZE)
    d_loss, real_score, fake_score, acc = update_D(real, fake)

    if frame_idx % 100 == 0:
        print('Iter: %d, mle_loss: %.3f, oracle_loss: %.3f, d_loss: %.3f, r_score: %.3f, f_score: %.3f, acc: %.3f' % (
            frame_idx, loss.item(), oracle_loss.item(), d_loss, real_score, fake_score, acc))

    return loss.item(), oracle_loss.item(), d_loss, real_score, fake_score, acc


if __name__ == '__main__':
    for frame_idx in count(1):
        if frame_idx < 15000:
            losses = pre_train(frame_idx)
            hard_update(G_t, G)
        else:
            losses = train_PG(frame_idx)
            soft_update(G_t, G)

        writer.add_scalars('SeqGAN', {'mse_loss': losses[0], 'oracle_loss': losses[1], 'd_loss': losses[2],
                                      'real_score': losses[3], 'fake_score': losses[4], 'acc': losses[5]}, frame_idx)
