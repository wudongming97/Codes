import torch.nn as nn
import torch.optim as optim

from data import mnist_train_iter, mnist_test_iter
from model import Net
from utils import *

n_epochs = 10
lr = 2e-4
seed = 77

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = Net()
print_network(model)

trainer = optim.Adam(model.parameters(), lr=lr, betas=[0.5, 0.99])
ce_criterion = nn.CrossEntropyLoss()
kl_criterion = nn.KLDivLoss()
log_softmax = nn.LogSoftmax(dim=1)

# train
for e in range(n_epochs):
    model.train()
    for b, (x, l) in enumerate(mnist_train_iter):
        x = x.view(-1, 784).to(DEVICE)
        l = l.to(DEVICE)

        rand_labels = torch.randint(0, 10, l.size(), dtype=torch.long).to(DEVICE)
        one_hot_rand_labels = one_hot(rand_labels.cpu(), 10).to(DEVICE)

        logits = model(x)
        loss = None  # TODO

        model.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % 50 == 0:
            acc = get_cls_accuracy(logits, l)
            print('[%2d/%2d] loss: %.3f acc: %.3f' % (e + 1, n_epochs, loss.item(), acc.item()))

    # valid
    with torch.no_grad():
        model.eval()
        total_acc = 0
        for x, l in mnist_test_iter:
            x = x.view(-1, 784).to(DEVICE)
            l = l.to(DEVICE)
            logits = model(x)
            total_acc += get_cls_accuracy(logits, l).item()
        acc = total_acc / len(mnist_test_iter)
        print('在测试集上的准确率为：%.3f ' % acc)
