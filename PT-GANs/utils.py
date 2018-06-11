import matplotlib.pyplot as plt
import numpy as np
import torch

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

seed = 77
torch.manual_seed(seed)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = torch.max(score, dim=1)
    correct = torch.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy


def to_categorical(labels):
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def one_hot(labels, num_classes):
    labels = labels.reshape(-1, 1)
    return (labels == torch.arange(num_classes).reshape(1, num_classes).long()).float()


def one_hot_v2(labels, num_classes):
    ones = torch.sparse.torch.eye(num_classes)
    return ones.index_select(0, labels)


def plot_q_z(x, y, filename):
    from sklearn.manifold import TSNE
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]

    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    if x.shape[1] != 2:
        x = TSNE().fit_transform(x)
    y = y[:, np.newaxis]
    xy = np.concatenate((x, y), axis=1)

    for l, c in zip(range(10), colors):
        ix = np.where(xy[:, 2] == l)
        ax.scatter(xy[ix, 0], xy[ix, 1], c=c, marker='o', label=l, s=10, linewidths=0)

    plt.savefig(filename)
    plt.close()
