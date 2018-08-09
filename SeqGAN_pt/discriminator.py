import torch
import torch.nn as nn

from utils import DEVICE


class Discriminator(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, max_seq_len=20, dropout_prob=0.1):
        super(Discriminator, self).__init__()
        self.max_seq_len = max_seq_len
        self.hid_size = hid_size
        self.embeddings = nn.Embedding(voc_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_size, 1),
            nn.Dropout(p=dropout_prob),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, x):
        emb = self.embeddings(x)
        hid = self.gru(emb)[1].view(-1, self.hid_size)
        return self.fc_out(hid)

    def rewards(self, rolls):
        rets = []
        for r in rolls:
            ret_t = self.forward(r)
            rets.append(ret_t)
        return torch.cat(rets, 1)
