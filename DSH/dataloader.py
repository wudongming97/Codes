import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image


def _file_paths(dir, ext):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith(ext):
            paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path, ext))
    return paths


class ImageFolder(data.Dataset):
    def __init__(self, root, ext='.png', is_training=True, transform=None):
        self.is_training = is_training
        self.sorted_paths = sorted(_file_paths(root, ext))
        self.L = len(self.sorted_paths) // 2
        self.paired_paths = [(self.sorted_paths[2 * l], self.sorted_paths[2 * l + 1]) for l in range(self.L)]
        self.transform = transform

    def __len__(self):
        return self.L

    def __getitem__(self, item):
        if self.is_training:
            rand_1, rand_2 = np.random.choice(range(self.L), 2, False)
            tuple_1, tuple_2 = self.paired_paths[rand_1], self.paired_paths[rand_2]

            pos_1 = Image.open(tuple_1[0]).convert("L")
            neg_1 = Image.open(tuple_1[1]).convert("L")
            pos_2 = Image.open(tuple_2[0]).convert("L")
            neg_2 = Image.open(tuple_2[1]).convert("L")

            if self.transform is not None:
                pos_1 = self.transform(pos_1)
                neg_1 = self.transform(neg_1)
                pos_2 = self.transform(pos_2)
                neg_2 = self.transform(neg_2)
            return pos_1, neg_1, pos_2, neg_2
        else:
            paths = self.paired_paths[item]
            pos = Image.open(paths[0]).convert("L")
            neg = Image.open(paths[0]).convert("L")
            if self.transform is not None:
                pos = self.transform(pos)
                neg = self.transform(neg)
            return pos, neg


_transformer = tv.transforms.Compose([
    tv.transforms.Resize([512, 512]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_iter = torch.utils.data.DataLoader(
    dataset=ImageFolder('../../Datasets/Dicom512_train/', transform=_transformer),
    batch_size=20,
    shuffle=True,
    drop_last=True
)

test_iter = torch.utils.data.DataLoader(
    dataset=ImageFolder('../../Datasets/Dicom512_test/', is_training=False, transform=_transformer),
    batch_size=10,
    drop_last=True
)
