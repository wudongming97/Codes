import torch
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
    def __init__(self, voc_size, emb_size, hid_size, n_layers, embedding=None):
        super(NaiveDecoder, self).__init__()
        self.n_layers = n_layers
        if embedding is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(emb_size, hid_size, n_layers)
        self.out = nn.Linear(hid_size, voc_size)
        self.to(DEVICE)

    def forward(self, x, hidden=None, dummy=None):
        embedd = self.embedding(x)
        output, hidden = self.gru(embedd, hidden)
        log_prob = F.log_softmax(self.out(output), -1)
        return log_prob, hidden, None


class BahdanauAttn(nn.Module):
    def __init__(self, hid_size):
        super(BahdanauAttn, self).__init__()
        self.hid_size = hid_size
        self.attn = nn.Sequential(
            nn.Linear(hid_size * 2, hid_size),
            nn.Tanh()
        )
        self.v = nn.Parameter(torch.ones(hid_size))
        nn.init.normal_(self.v, 0.02)

    def score(self, ST, HT):
        energy = self.attn(torch.cat([HT, ST], -1)).transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(HT.size(1), 1).unsqueeze(1)  # # [B*1*H]
        return (v @ energy).squeeze(1)  # [B*T]

    def forward(self, St, HT):
        HT = HT.transpose(0, 1)
        ST = St.repeat(HT.size(0), 1, 1).transpose(0, 1)  # [B*T*H]
        attn_weight = F.softmax(self.score(ST, HT), -1).unsqueeze(1)  # [B*1*T]
        context = (attn_weight @ HT).transpose(0, 1)  # (1,B,N)
        return context, attn_weight


class BahdanauAttnDecoder(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, n_layers, embedding=None):
        super(BahdanauAttnDecoder, self).__init__()
        self.n_layers = n_layers
        if embedding is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(emb_size + hid_size, hid_size, n_layers)
        self.attn = BahdanauAttn(hid_size)
        self.out = nn.Linear(hid_size, voc_size)
        self.to(DEVICE)

    def forward(self, x, latest_hid, HT):
        embedd = self.embedding(x)
        context, attn_weight = self.attn(latest_hid[-1], HT)
        rnn_inp = torch.cat([embedd, context], -1)
        output, hidden = self.gru(rnn_inp, latest_hid)
        log_prob = F.log_softmax(self.out(output), -1)
        return log_prob, hidden, attn_weight


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec
        self.to(DEVICE)
        print(self)

    def forward(self, src, trg):
        # 完全teach force
        enc_out, hidden = self.enc(src)
        hidden = hidden[-self.dec.n_layers:, :, :]

        outputs = []
        max_seq_len = trg.size(0)
        for t in range(max_seq_len):
            input = trg[t:t + 1, :]
            log_prob, hidden, attn_weight = self.dec(input, hidden, enc_out)
            outputs.append(log_prob)

        return torch.cat(outputs)
