import torch
import torch.nn as nn
import torch.utils.data as data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleTensorDataset(data.Dataset):
    def __init__(self, file):
        self.tensor = torch.load(file)

    def __getitem__(self, item):
        return self.tensor[item]

    def __len__(self):
        return self.tensor.size(0)


def data_iter(file, batch_size):
    return data.DataLoader(
        dataset=SingleTensorDataset(file=file),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=2,
    )


def prepare_gru_inputs(input, sos=0):
    """
    :param input: batch_size x seq_len
    :param sos:
    :return:
    """
    inp = torch.ones_like(input) * sos
    inp[:, 1:] = input[:, :input.size(1) - 1]
    return inp


def nll_loss(oracle_net, target):
    inp = prepare_gru_inputs(target, oracle_net.sos)
    out, _ = oracle_net(inp)
    nll = nn.NLLLoss(reduction='elementwise_mean')
    return nll(out.view(-1, out.size(-1)), target.view(-1))


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
