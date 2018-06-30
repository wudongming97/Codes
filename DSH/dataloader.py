import os
import random

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
        rand_num_1 = random.choice(range(self.L))
        tuple_paths_1 = self.paired_paths[rand_num_1]

        pos_img_1 = Image.open(tuple_paths_1[0]).convert("L")
        pos_img_2 = Image.open(tuple_paths_1[1]).convert("L")
        if self.transform is not None:
            pos_img_1 = self.transform(pos_img_1)
            pos_img_2 = self.transform(pos_img_2)

        if self.is_training:
            rand_num_2 = random.choice(range(self.L * 2))
            while rand_num_1 * 2 == rand_num_2 or rand_num_1 * 2 + 1 == rand_num_2:
                rand_num_2 = random.choice(range(self.L * 2))
            neg_img = Image.open(self.sorted_paths[rand_num_2]).convert("L")
            if self.transform is not None:
                neg_img = self.transform(neg_img)
            return pos_img_1, pos_img_2, neg_img
        else:
            return pos_img_1, pos_img_2


_transformer = tv.transforms.Compose([
    tv.transforms.Resize([512, 512]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_iter = torch.utils.data.DataLoader(
    dataset=ImageFolder('../../Datasets/Dicom512_train/', transform=_transformer),
    batch_size=10,
    shuffle=True,
    drop_last=True
)

test_iter = torch.utils.data.DataLoader(
    dataset=ImageFolder('../../Datasets/Dicom512_test/', is_training=False, transform=_transformer),
    batch_size=10,
    drop_last=True
)
