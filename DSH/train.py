import os

import torch.nn.functional as F
import torch.optim as optim

from dataloader import train_iter, test_iter
from network import get_network
from test import test
from utils import *

m = 4
alpha = 0.001

lr = 2e-4
n_epochs = 100
os.makedirs(save_dir, exist_ok=True)

model = get_network().to(DEVICE)
print_network(model)
trainer = optim.Adam(model.parameters(), lr=lr, betas=[0.5, 0.999])

for epoch in range(n_epochs):
    model.train()
    for b, (pos_1, neg_1, pos_2, neg_2) in enumerate(train_iter):
        pos_1 = pos_1.to(DEVICE)
        neg_1 = neg_1.to(DEVICE)
        pos_2 = pos_2.to(DEVICE)
        neg_2 = neg_2.to(DEVICE)

        # 正向
        pb_1 = model(pos_1)
        nb_1 = model(neg_1, False)
        pb_2 = model(pos_2)
        nb_2 = model(neg_2, False)

        # pos+neg
        loss = ((pb_1 - nb_1) ** 2).sum(1).mean(0)
        # loss += alpha * ((pb_1.abs() - 1).abs() + (nb_1.abs() - 1).abs()).sum(1).mean(0)
        loss += ((pb_2 - nb_2) ** 2).sum(1).mean(0)
        # loss += alpha * ((pb_2.abs() - 1).abs() + (nb_2.abs() - 1).abs()).sum(1).mean(0)

        # 交错
        loss += F.relu(m - ((pb_1 - nb_2) ** 2).sum(1).mean(0))
        # loss += alpha * ((pb_1.abs() - 1).abs() + (nb_2.abs() - 1).abs()).sum(1).mean(0)
        loss += F.relu(m - ((pb_2 - nb_1) ** 2).sum(1).mean(0))
        # loss += alpha * ((pb_2.abs() - 1).abs() + (nb_1.abs() - 1).abs()).sum(1).mean(0)

        # 同错
        loss += F.relu(m - ((pb_1 - pb_2) ** 2).sum(1).mean(0))
        # loss += alpha * ((pb_1.abs() - 1).abs() + (pb_2.abs() - 1).abs()).sum(1).mean(0)
        loss += F.relu(m - ((nb_1 - nb_2) ** 2).sum(1).mean(0))
        # loss += alpha * ((nb_1.abs() - 1).abs() + (nb_2.abs() - 1).abs()).sum(1).mean(0)

        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % 5 == 0:
            print('[%3d/%3d] [%3d] loss: %.6f' % (epoch + 1, n_epochs, b + 1, loss.item()))

    # save
    # torch.save(model.state_dict(), save_dir + '%d.pth' % (epoch + 1))
    # test
    top_k = test(model, train_iter)
    print('[Train] top_k: %.3f' % (sum(top_k) / len(top_k)))
    top_k = test(model, test_iter)
    print('[Test] top_k: %.3f' % (sum(top_k) / len(top_k)))
