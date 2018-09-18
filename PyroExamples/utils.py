import torch
import torch.utils.data as data
import torchvision as tv

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')


def anime_face_loader(root, transform, batch_size=128):
    face_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.ImageFolder(root=root, transform=transform),
        batch_size=batch_size, shuffle=True)
    return face_loader


def mnist_loaders(root, batch_size=128):
    trans = tv.transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=True, transform=trans, download=True),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=tv.datasets.MNIST(root=root, train=False, transform=trans),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = torch.max(score, dim=1)
    correct = torch.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy
