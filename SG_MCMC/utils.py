import torch
import torchvision as tv

seed = 77
torch.manual_seed(seed)

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


def mnist_loaders(root, batch_size=128):
    trans = tv.transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=True, transform=trans, download=True),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=False, transform=trans),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
