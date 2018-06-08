import torch
import torchvision as tv

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = torch.max(score, dim=1)
    correct = torch.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy


mnist_train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

mnist_test_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=100,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
