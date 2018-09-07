import torch

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

_seed = 77
torch.manual_seed(_seed)

save_dir = './Results/'


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def mean_topk(model, parallel_iter, monolingual_iter=None):
    """
    有肉的在前面，为pos，骨架为neg
    :param model:
    :param parallel_iter:
    :param monolingual_iter:
    :return:
    """
    with torch.no_grad():
        model.eval()
        pb = []
        nb = []
        for pos, neg, _, _ in parallel_iter:
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)
            pb.append(model(pos))
            nb.append(model(neg, False))
        if monolingual_iter is not None:
            for pos in monolingual_iter:
                pos = pos.to(DEVICE)
                pb.append(model(pos))

        pb = torch.cat(pb)
        nb = torch.cat(nb)

        pb_len = pb.size(0)
        nb_len = nb.size(0)
        distance_mx = torch.zeros(nb_len, pb_len, dtype=torch.float, device=DEVICE)
        for i in range(nb_len):
            for j in range(pb_len):
                distance_mx[i][j] = ((nb[i] - pb[j]) ** 2).mean()

        top_k = [0] * nb_len
        for i in range(nb_len):
            for j in range(pb_len):
                if distance_mx[i][j] < distance_mx[i][i]:
                    top_k[i] += 1

        return top_k
