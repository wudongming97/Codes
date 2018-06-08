import torch.nn as nn
import torch.optim as optim

from utils import *


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = std.new_tensor(torch.randn(std.size()))
    return mu + std * eps


class VIB(nn.Module):
    def __init__(self, z_dim=256):
        super(VIB, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2)
        )
        self.decoder = nn.Linear(z_dim, 10)
        self.to(DEVICE)

    def forward(self, x):
        dist = self.encoder(x)
        mu = dist[:, :self.z_dim]
        logvar = dist[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar


z_dim = 256
lr = 2e-4
n_epochs = 10
beta = 1e-3
display_interval = 50

model = VIB(z_dim)
print_network(model)
trainer = optim.Adam(model.parameters(), lr, [0.5, 0.99])
ce_criterion = nn.CrossEntropyLoss(size_average=False)

for e in range(n_epochs):
    model.train()
    for b, (x, l) in enumerate(mnist_train_iter):
        x = x.view(-1, 784).to(DEVICE)
        l = l.to(DEVICE)

        logits, mu, logvar = model(x)
        ce_loss = ce_criterion(logits, l) / l.size(0)
        kl_loss = -0.5 * (logvar - mu ** 2 + 1 - logvar.exp()).sum(1).mean()
        loss = ce_loss + beta * kl_loss

        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % display_interval == 0:
            acc = get_cls_accuracy(logits, l)
            print('[ %d / %d ] acc: %.4f ce_loss: %4f kl_loss: %.4f' % (
                e + 1, n_epochs, acc.item(), ce_loss.item(), kl_loss.item()))

    with torch.no_grad():
        model.eval()
        total_acc = 0
        for x, l in mnist_test_iter:
            x = x.view(-1, 784).to(DEVICE)
            l = l.to(DEVICE)

            logits, _, _ = model(x)
            total_acc += get_cls_accuracy(logits, l).item()
        acc = total_acc / len(mnist_test_iter)
        print('在测试集上的准确率为：%.3f ' % acc)
