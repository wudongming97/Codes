import torch.nn.functional as F
import torch.nn.init as init

from utils import *


class Generator(nn.Module):
    def __init__(self, voc_size, emb_size, hid_size, max_seq_len=20, sos=0, oracle_init=False):
        super(Generator, self).__init__()
        self.max_seq_len = max_seq_len
        self.sos = sos

        self.embedding = nn.Embedding(voc_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.fc_out = nn.Linear(hid_size, voc_size)

        if oracle_init:
            for p in self.parameters():
                init.normal_(p, 0, 1)

        self.to(DEVICE)

    def forward(self, input, hidden=None):
        emb = self.embedding(input)
        out, hidden = self.gru(emb, hidden)
        out = self.fc_out(out)
        out = F.log_softmax(out, -1)

        return out, hidden

    def sample(self, input, seq_len):
        assert seq_len > 0
        output = []
        for t in range(seq_len):
            out = self.forward(input)[0].squeeze()
            input = torch.multinomial(torch.exp(out), 1)
            output.append(input)
        return torch.cat(output, -1)

    def log_pt(self, input):
        """
        :param G: 生成器
        :param input: batch_size x seq_len
        :return:
        """
        batch_size, seq_len = input.size()
        inp = prepare_gru_inputs(input, self.sos)
        out = self(inp.to(DEVICE))[0]
        out = out.view(-1, out.size(-1))
        log_probs = out.gather(-1, input.view(-1, 1))
        return log_probs.view(batch_size, seq_len)

    def roll_out(self, input):
        """
        :param input: batch_size x seq_len
        :return:
        """
        seq_len = input.size(1)
        out = []
        for t in range(seq_len - 1):
            inp = input[:, t]
            remains = self.sample(inp.unsqueeze(-1), seq_len - t - 1)
            res = torch.cat([input[:, :t + 1], remains], 1)
            out.append(res)
        out.append(input)
        return out
