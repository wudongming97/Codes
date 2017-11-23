import torch
from torch.autograd import Variable


def cast(var, type):
    """ Cast a Tensor to the given type.
        ```
        input = torch.FloatTensor(1)
        target_type = type(torch.LongTensor(1))
        type(torch.cast(input, target_type))
        # <class 'torch.LongTensor'>
        ```
    """
    if type == torch.ByteTensor:
        return var.byte()
    elif type == torch.CharTensor:
        return var.char()
    elif type == torch.DoubleTensor:
        return var.double()
    elif type == torch.FloatTensor:
        return var.float()
    elif type == torch.IntTensor:
        return var.int()
    elif type == torch.LongTensor:
        return var.long()
    elif type == torch.ShortTensor:
        return var.short()
    else:
        raise ValueError("Not a Tensor type.")


def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
    """
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret


def nll(log_prob, label):
    """ Is similar to [`nll_loss`](http://pytorch.org/docs/nn.html?highlight=nll#torch.nn.functional.nll_loss) except does not return an aggregate.
        ```
        input = Variable(torch.FloatTensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]]))
        target = Variable(torch.LongTensor([1, 2]).view(-1, 1))
        output = torch.nll(torch.log(input), target)
        output.size()
        # (2,)
        ```
    """
    if isinstance(log_prob, Variable):
        _type = type(log_prob.data)
    else:
        _type = type(log_prob)

    mask = one_hot(log_prob.size(), label)
    mask = cast(mask, _type)
    return -1 * (log_prob * mask).sum(1)


if __name__ == '__main__':
    # test one_hot
    size = (3,4)
    index = torch.LongTensor([0, 1, 2]).view(-1,1)
    out = one_hot(size, index)
    print(out)

