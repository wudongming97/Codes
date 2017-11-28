import torch
from torch.autograd import Variable

USE_GPU = torch.cuda.is_available()


def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
    size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    """
    mask = torch.LongTensor(*size).fill_(0)
    ret = mask.scatter_(1, index, 1)
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
    mask = Variable(one_hot(log_prob.size(), label.data.cpu()).type_as(log_prob.data))
    if USE_GPU:
        mask = mask.cuda()
    # FloatTensor 不能跟 LongTensor 相乘，？？？？？
    return -1 * (log_prob * mask).sum(1)


def print_sentences(sentences):
    for s in sentences:
        print('  ' + s)


if __name__ == '__main__':
    # test one_hot
    size = (3,4)
    index = torch.LongTensor([0, 1, 2]).view(-1,1)
    out = one_hot(size, index)
    print(out)

