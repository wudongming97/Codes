import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.log_normal import LogNormal

from utils import *


class InformationDropout(nn.Module):
    def __init__(self, in_featurs, out_featurs, max_apha):
        super(InformationDropout, self).__init__()
        self.max_alpha = max_apha
        self.fx = nn.Sequential(
            nn.Linear(in_featurs, out_featurs),
            nn.ReLU(True),
        )
        self.alpha = nn.Sequential(
            nn.Linear(in_featurs, out_featurs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        fx = self.fx(x)
        alpha = 1e-3 + self.max_alpha * self.alpha(x)
        log_normal = LogNormal(torch.zeros_like(fx), alpha)
        kl = -torch.log(alpha / (self.max_alpha + 1e-3))
        eps = log_normal.sample()
        return fx * eps, kl


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.info_drop1 = InformationDropout(1024, 1024, 0.7)
        self.fc2 = nn.Linear(1024, 512)
        self.info_drop2 = InformationDropout(512, 128, 0.7)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        fc1 = F.relu(self.fc1(x))
        fx1, kl1 = self.info_drop1(fc1)
        fc2 = F.relu(self.fc2(fx1))
        fx2, kl2 = self.info_drop2(fc2)
        logits = self.fc3(fx2)

        return logits, (kl1, kl2)


model = NET().to(DEVICE)
print_network(model)

lr = 2e-4
beta = 0.1
n_epochs = 10
display_interval = 20

trainer = optim.Adam(model.parameters(), lr=lr, betas=[0.5, 0.99])
ce_criterion = nn.CrossEntropyLoss()

for e in range(n_epochs):
    for b, (x, l) in enumerate(mnist_train_iter):
        x = x.view(-1, 784).to(DEVICE)
        l = l.to(DEVICE)
        logits, kls = model(x)
        ce_loss = ce_criterion(logits, l)
        loss = ce_loss
        for kl in kls:
            loss += beta * torch.mean(kl)

        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % display_interval == 0:
            acc = get_cls_accuracy(logits, l)
            print('[ %d / %d ] acc: %.4f ce_loss: %4f loss: %.4f' % (
                e + 1, n_epochs, acc.item(), ce_loss.item(), loss.item()))

    # test
    with torch.no_grad():
        model.eval()
        total_acc = 0
        for i, (x, l) in enumerate(mnist_test_iter):
            x = x.view(-1, 784).to(DEVICE)
            l = l.to(DEVICE)

            logits, _ = model(x)
            total_acc += get_cls_accuracy(logits, l).item()
        acc = total_acc / len(mnist_test_iter)
        print('在测试集上的准确率为：%.3f ' % acc)
