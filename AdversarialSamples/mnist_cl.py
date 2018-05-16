import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from utils import get_cls_accuracy

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

batch_size = 32

train_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

test_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=1000,
    shuffle=False,
    drop_last=True,
    num_workers=2,
)


class mnist_cl(nn.Module):
    def __init__(self):
        super(mnist_cl, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 7, 1, 0),
        )

    def forward(self, input):
        return self.net(input).squeeze()


criterion = nn.CrossEntropyLoss()


def train(model, data_iter, n_epochs=20, lr=1e-3, to_display=100):
    model.train()
    num_batchs = len(data_iter)
    opt = optim.Adam(model.parameters(), lr, betas=[0.5, 0.99])
    for epoch in range(n_epochs):
        for batch, (X, L) in enumerate(data_iter):
            logits = model(X.to(DEVICE))
            acc = get_cls_accuracy(logits, L.to(DEVICE))
            loss = criterion(logits, L.to(DEVICE))

            model.zero_grad()
            loss.backward()
            opt.step()

            if (batch + 1) % to_display == 0:
                print('[%3d/%3d] [%4d/%4d] loss: %.3f acc: %.3f ' % (
                    epoch + 1, n_epochs, batch + 1, num_batchs, loss.item(), acc.item()))


def valid(model, data_iter):
    model.eval()
    num_batchs = len(data_iter)
    loss, acc = 0, 0
    with T.no_grad():
        for batch, (X, L) in enumerate(data_iter):
            logits = model(X.to(DEVICE))
            acc += get_cls_accuracy(logits, L.to(DEVICE)).item()
            loss = criterion(logits, L.to(DEVICE)).item()
    return loss / num_batchs, acc / num_batchs


if __name__ == '__main__':
    model = mnist_cl().to(DEVICE)
    train(model, train_iter, n_epochs=3, lr=1e-4)
    loss, acc = valid(model, test_iter)
    print('valid: loss: %.3f acc: %.3f' % (loss, acc))
    T.save(model.state_dict(), 'mnist_cl.pt')
