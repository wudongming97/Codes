import os

import torch.nn.functional as F
import torch.optim as optim

from dataloader import data_iter, test_iter
from network import get_network
from utils import *

m = 4
alpha = 0.05

lr = 2e-4
n_epochs = 20
save_dir = './Results/'
os.makedirs(save_dir, exist_ok=True)

model = get_network().to(DEVICE)
print_network(model)
trainer = optim.Adam(model.parameters(), lr=lr, betas=[0.5, 0.999])

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

        if (b + 1) % 5 == 0:
            print('[%3d/%3d] [%3d] loss: %.6f' % (epoch + 1, n_epochs, b + 1, loss.item()))

    # save
    torch.save(model.state_dict(), save_dir + '%d.pth' % (epoch + 1))
    # test
    with torch.no_grad():
        model.eval()
        pb1 = []
        pb2 = []
        for pos1, pos2 in test_iter:
            pos1 = pos1.to(DEVICE)
            pos2 = pos2.to(DEVICE)
            pb1.append(model(pos1))
            pb2.append(model(pos2))

        pb1 = torch.cat(pb1)
        pb2 = torch.cat(pb2)

        pb_len = pb1.size(0)
        distance_mx = torch.zeros(pb_len, pb_len, dtype=torch.float, device=DEVICE)
        for i in range(pb_len):
            for j in range(pb_len):
                distance_mx[i][j] = ((pb1[i] - pb2[j]) ** 2).mean()

        top_k = [0] * pb_len
        for i in range(pb_len):
            for j in range(pb_len):
                if distance_mx[i][j] < distance_mx[i][i]:
                    top_k[i] += 1
        print('--' * 20)
        print('Epochs [%3d] top_k: %.3f' % (epoch + 1, sum(top_k) / len(top_k)))
        print(",".join(str(i) for i in top_k))
        print('--' * 20)
