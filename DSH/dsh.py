import os

import torch.nn.functional as F
import torch.optim as optim

from dataloader import data_iter
from networks import dsh_network
from utils import *

nf = 64
lr = 1e-3
input_nc = 1
save_dir = './Results/'
os.makedirs(save_dir, exist_ok=True)

model = dsh_network(input_nc, nf).to(DEVICE)
print_network(model)
trainer = optim.Adam(model.parameters(), lr=lr, betas=[0.5, 0.999])

alpha = 0.05
m = 3
n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    for b, (pos_1, pos_2, neg) in enumerate(data_iter):
        pos_1 = pos_1.to(DEVICE)
        pos_2 = pos_2.to(DEVICE)
        neg = neg.to(DEVICE)

        pb_1 = model(pos_1)
        pb_2 = model(pos_2)
        nb = model(neg)

        # pos1+pos2
        loss = ((pb_1 - pb_2) ** 2).sum(1).mean(0)
        loss += alpha * ((pb_1.abs() - 1).abs() + (pb_2.abs() - 1).abs()).sum(1).mean(0)

        # pos1+neg
        loss += F.relu(m - ((pb_1 - nb) ** 2).sum(1).mean(0)) * 0.5
        loss += alpha * ((pb_1.abs() - 1).abs() + (nb.abs() - 1).abs()).sum(1).mean(0)

        # pos2+neg
        loss += F.relu(m - ((pb_2 - nb) ** 2).sum(1).mean(0)) * 0.5
        loss += alpha * ((pb_2.abs() - 1).abs() + (nb.abs() - 1).abs()).sum(1).mean(0)

        trainer.zero_grad()
        loss.backward()
        trainer.step()

        if (b + 1) % 20 == 0:
            print('[%3d/%3d] [%3d] loss: %.6f' % (epoch + 1, n_epochs, b + 1, loss.item()))

    # save
    torch.save(model.state_dict(), save_dir + '%d.pth' % (epoch + 1))
    # test
    with torch.no_grad():
        model.eval()

