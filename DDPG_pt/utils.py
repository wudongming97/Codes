import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(var):
    return var.cpu().data.numpy()


def to_tensors(v_list):
    return list(map(lambda x: torch.tensor(x).float().to(DEVICE), v_list))


def to_tensor(var):
    return torch.tensor(var).float().to(DEVICE)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
