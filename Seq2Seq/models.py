import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


def length_norm(lengths, alpha=0.7):
    return (5 + 1) / (5 + lengths ** alpha)


class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, n_layers, embedding=None):
        super(Encoder, self).__init__()
        self.hid_size = hid_size
        if embedding is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.embedding = embedding
        self.gru = nn.GRU(emb_size, hid_size, n_layers, bidirectional=True)
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
        v = self.v.repeat(HT.size(0), 1).unsqueeze(1)  # # [B*1*H]
        return v @ energy  # [B*1*T]

    def forward(self, St, HT):
        ST = St.repeat(HT.size(0), 1, 1).transpose(0, 1)  # [B*T*H]
        HT = HT.transpose(0, 1)  # [B*T*H]
        attn_weight = F.softmax(self.score(ST, HT), -1)
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
        context, attn_weight = self.attn(latest_hid[-1:], HT)
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

    def rand_sample(self, src, sos_idx, eos_idx, max_len=50):
        assert src.size(1) == 1  # 每次只采样一个sample
        enc_out, hidden = self.enc(src)
        hidden = hidden[-self.dec.n_layers:, :, :]

        output = []
        input = sos_idx * torch.ones(1, 1).long().to(DEVICE)
        for t in range(max_len):
            log_prob, hidden, attn_weight = self.dec(input, hidden, enc_out)
            word_idx = log_prob.max(-1)[1].item()
            if word_idx == eos_idx:
                break
            output.append(word_idx)
        return torch.tensor(output)

    def beam_sample(self, src, sos_idx, eos_idx, max_len=30, k=10):
        assert src.size(1) == 1  # 每次只采样一个sample
        outputs, mask, score, length = None, None, None, None

        enc_out, hidden = self.enc(src)
        hidden = hidden[-self.dec.n_layers:, :, :]
        input = sos_idx * torch.ones(1, 1).long().to(DEVICE)
        for t in range(max_len):
            rnn_out, hidden, _ = self.dec(input, hidden, enc_out)
            rnn_out = rnn_out.squeeze(0)
            val, idx = rnn_out.topk(k, dim=-1, sorted=False)
            if mask is not None and mask.sum() == 0:
                break
            if t == 0:
                score = val
                outputs = idx.t()
                mask = (idx != eos_idx)
                length = mask
                hidden = hidden.repeat(1, k, 1)
                enc_out = enc_out.repeat(1, k, 1)
            else:
                pre_score = score.t().repeat(1, k)  # kxk
                pre_mask = mask.t().repeat(1, k)
                pre_length = length.t().repeat(1, k)
                length = (pre_length + (idx != eos_idx)).view(1, -1)
                cur_score = (pre_score + val * pre_mask.float()).view(1, -1)
                lp_score = cur_score * length_norm(length).float()
                _, score_idx = lp_score.topk(k, sorted=False)
                score = cur_score.gather(1, score_idx)
                length = pre_length.view(1, -1).gather(1, score_idx)
                selected_idx = idx.view(1, -1).gather(1, score_idx)
                mask = pre_mask.view(1, -1).gather(1, score_idx) * (selected_idx != eos_idx)
                pre_idx = (score_idx / k).long().squeeze()
                outputs = torch.cat([outputs.index_select(0, pre_idx), selected_idx.t()], -1)

            input = outputs[:, -1].unsqueeze(0)

        # 选取最大的一个
        best_score, ix = score.topk(1)
        return outputs[ix.item()], best_score
