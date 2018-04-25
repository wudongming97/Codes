import os
import random

import torch as T
import torch.utils.data as data
import torchvision as tv
from PIL import Image

_ds_root_dir = {
    'horse2zebra': '../../Datasets/coco/horse2zebra/',
    'apple2orange': '../../Datasets/coco/apple2orange/'
}


class single_class_image_folder(data.Dataset):
    def __init__(self, root, transform=None):
        self.img_paths = self._file_paths(root)
        self.L = len(self.img_paths)
        self.transform = transform

    @staticmethod
    def _file_paths(dir):
        paths = []
        for img in os.scandir(dir):
            paths.append(img.path)
        return paths

    def __len__(self):
        return self.L

    def __getitem__(self, item):
        f_path = self.img_paths[random.choice(range(self.L))]
        img = Image.open(f_path)
        if self.transform is not None:
            img = self.transform(img)
        return img


class paired_image_dataset(data.Dataset):
    def __init__(self, paired_paths, transform):
        self.paired_paths = paired_paths
        self.transform = transform

    def __getitem__(self, item):
        paired_path = self.paired_paths[item]
        im1 = Image.open(paired_path[0])
        im2 = Image.open(paired_path[1])
        if self.transform is not None:
            im1 = self.transform(im1).convert('RGB')
            im2 = self.transform(im2).convert('RGB')
        return im1, im2

    def __len__(self):
        return len(self.paired_paths)


class random_paired_dataset(data.Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds1_l = len(self.ds1)
        self.ds2 = ds2
        self.ds2_l = len(self.ds2)

    def __len__(self):
        return min(self.ds1_l, self.ds2_l)

    def __getitem__(self, item):
        d1 = self.ds1[random.choice(range(self.ds1_l))]
        d2 = self.ds2[random.choice(range(self.ds1_l))]
        return d1, d2


_transformer = tv.transforms.Compose([
    tv.transforms.Resize([256, 256]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

horse2zebra_loader = T.utils.data.DataLoader(
    dataset=random_paired_dataset(
        single_class_image_folder(_ds_root_dir['horse2zebra'] + 'trainA/', _transformer),
        single_class_image_folder(_ds_root_dir['horse2zebra'] + 'trainB/', _transformer)),
    batch_size=1, drop_last=True, num_workers=2)

apple2orange_loader = T.utils.data.DataLoader(
    dataset=random_paired_dataset(
        single_class_image_folder(_ds_root_dir['apple2orange'] + 'trainA/', _transformer),
        single_class_image_folder(_ds_root_dir['apple2orange'] + 'trainB/', _transformer)),
    batch_size=1, drop_last=True, num_workers=2)
