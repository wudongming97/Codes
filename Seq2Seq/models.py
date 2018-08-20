import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, n_layers, drop_prob=0.5, embedding=None):
        super(Encoder, self).__init__()
        self.hid_size = hid_size
        if embedding is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(emb_size, hid_size, n_layers, dropout=drop_prob, bidirectional=True)
        self.to(DEVICE)

    def forward(self, x):
        embedd = self.embedding(x)
        output, hidden = self.gru(embedd)
        output = output[:, :, :self.hid_size] + output[:, :, self.hid_size:]
        return output, hidden


class NaiveDecoder(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, n_layers, drop_prob=0.5, embedding=None):
        super(NaiveDecoder, self).__init__()
        self.n_layers = n_layers
        if embedding is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(emb_size, hid_size, n_layers, dropout=drop_prob)
        self.out = nn.Linear(hid_size, voc_size)
        self.to(DEVICE)

    def forward(self, x, hidden=None):
        embedd = self.embedding(x)
        output, hidden = self.gru(embedd, hidden)
        log_prob = F.log_softmax(self.out(output), -1)
        return log_prob, hidden


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec
        self.to(DEVICE)
        print(self)

    def forward(self, src, trg):
        """
        :param src:
        :param trg:
        :return:
        """
        _, hidden = self.enc(src)
        return self.dec(trg, hidden[-self.dec.n_layers:, :, :])[0]
