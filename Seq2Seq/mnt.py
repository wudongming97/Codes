import os

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models import *
from utils import *


class MNT:
    def __init__(self, save_dir, src, trg, tri_iter, val_iter, tst_iter,
                 e_emb_size=512, e_hid_size=512, e_n_layers=1,
                 d_emb_size=512, d_hid_size=512, d_n_layers=1,
                 lr=1e-3, n_epochs=20, grad_clip=5, print_every=100, use_attn=False):

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.lr = lr
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.print_every = print_every

        self.e_vocab_size = len(src.vocab)
        self.d_vocab_size = len(trg.vocab)
        self.tri_iter = tri_iter
        self.val_iter = val_iter
        self.tst_iter = tst_iter

        enc = Encoder(self.e_vocab_size, e_emb_size, e_hid_size, e_n_layers)
        if use_attn:
            dec = BahdanauAttnDecoder(self.d_vocab_size, d_emb_size, d_hid_size, d_n_layers)
        else:
            dec = NaiveDecoder(self.d_vocab_size, d_emb_size, d_hid_size, d_n_layers)
        self.seq2seq = Seq2Seq(enc, dec)

        print('--' * 10)
        print(self.seq2seq)

        self.criterion = nn.NLLLoss(ignore_index=trg.vocab.stoi['<pad>'])
        self.optimizer = optim.Adam(self.seq2seq.parameters(), lr=lr, betas=(0.5, 0.999))

    def valid(self):
        self.seq2seq.eval()
        total_loss = 0
        with T.no_grad():
            for batch in self.val_iter:
                outputs = self.seq2seq(batch.src, batch.trg[:-1])
                loss = self.criterion(outputs.view(-1, self.d_vocab_size), batch.trg[1:].contiguous().view(-1))
                total_loss += loss.item()
        return total_loss / len(self.val_iter)

    def fit(self):
        for epoch in range(self.n_epochs):
            print('--' * 10)
            # train
            for b, batch in enumerate(self.tri_iter):
                self.seq2seq.train()
                outputs = self.seq2seq(batch.src, batch.trg[:-1])
                loss = self.criterion(outputs.view(-1, self.d_vocab_size), batch.trg[1:].contiguous().view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.seq2seq.parameters(), self.grad_clip)
                self.optimizer.step()
                if b % self.print_every == 0:
                    val_loss = self.valid()
                    print("[ %3d/%5d ] [ train_loss: %5.2f ] [valid_loss: %5.2f] " % (
                        epoch, (b + 1), loss.item(), val_loss))

            test(self.seq2seq, self.tri_iter, ('train_1.txt', 'train_2.txt'))
            test(self.seq2seq, self.tst_iter, ('test_1.txt', 'test_2.txt'))
