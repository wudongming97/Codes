import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from network import Conv2dBlock, STNLayer
from utils import *


# dataset
def _file_paths(dir, ext):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith(ext):
            if entry.path[-5] == 'R':
                paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path, ext))
    return paths


class ImageFolder(Dataset):
    def __init__(self, root, ext='.png', is_train=True, transform=None, test_rate=0.2):
        self.transform = transform
        self.is_train = is_train
        self.sorted_paths = sorted(_file_paths(root, ext))
        self.L = len(self.sorted_paths)
        self.train_len = int(np.ceil((1 - test_rate) * self.L))
        self.test_len = int(np.floor(test_rate * self.L))
        self.train_paths = self.sorted_paths[:int(self.train_len)]
        self.test_paths = self.sorted_paths[int(self.train_len):]
        print('数据总量：{}， 其中{}条用于训练{}条用于测试........................'.format(self.L, self.train_len, self.test_len))

    def __len__(self):
        if self.is_train == True:
            return self.train_len
        else:
            return self.test_len

    def __getitem__(self, item):
        src_dir = '/home/yx/Datasets/resize_Dicom512/'
        trg_data_paths = self.test_paths
        if self.is_train:
            trg_data_paths = self.train_paths

        tail = trg_data_paths[item][34:]
        src_data_path = src_dir + tail
        train_data = Image.open(src_data_path).convert('L')
        train_label = Image.open(trg_data_paths[item]).convert('L')
        if self.transform is not None:
            train_data = self.transform(train_data)
            train_label = self.transform(train_label)
        return train_data, train_label


_transformer = transforms.Compose([
    transforms.Grayscale(),
    tv.transforms.Resize([256, 256]),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = ImageFolder(root=r'/home/yx/Datasets/resize_shougong', ext='.png', is_train=True,
                            transform=_transformer, test_rate=0.2)
test_dataset = ImageFolder(root=r'/home/yx/Datasets/resize_shougong', ext='.png', is_train=False,
                           transform=_transformer, test_rate=0.2)

# Training dataset
train_loader = DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, )
# Test dataset
test_loader = DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True, num_workers=4)

model = nn.Sequential(
    STNLayer(1),
    Conv2dBlock(1, 32, 3, 1, 1),
    STNLayer(32),
    nn.Conv2d(32, 1, 3, 1, 1),
    nn.Sigmoid()
).to(DEVICE)

trainer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        trainer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        # loss = F.smooth_l1_loss(output, target)
        loss.backward()
        trainer.step()

        if batch_idx % 5 == 0:
            print('[Train] Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data) if (batch_idx + 1) * 64 <= len(train_loader.dataset) else len(
                    train_loader.dataset),
                len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

        save_imgs = torch.cat([output.detach()[:4], target[:4]], 0)
        tv.utils.save_image(save_imgs, save_dir + '{}_{}.png'.format(epoch, batch_idx), 4)
        torch.save(model.state_dict(), 'stn_{}.pth'.format(epoch))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
