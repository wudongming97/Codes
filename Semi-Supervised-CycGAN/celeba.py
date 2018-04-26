import random

import torch as T
import torch.utils.data as data
import torchvision as tv
from PIL import Image

batch_size = 4


class celeba_dataset(data.Dataset):
    def __init__(self, celeba_path, target_attr, transform=None):
        self.attr = target_attr
        self.root = celeba_path
        self.att_file = self.root + 'list_attr_celeba.txt'
        self.celeba_dir = self.root + 'img_align_celeba/'
        self.transform = transform
        self.pos_paths = []
        self.neg_paths = []
        self._paths_with_attr()

    def _paths_with_attr(self):
        with open(self.att_file) as f:
            attrs = [l for l in f]

        attr_id = 0
        for att_name in attrs[1].split():
            attr_id += 1
            if att_name == self.attr:
                break

        for l in attrs[2:]:
            lbls = l.split()
            if lbls[attr_id] == '1':
                self.pos_paths.append(self.celeba_dir + lbls[0])
            else:
                self.neg_paths.append(self.celeba_dir + lbls[0])
        self.pos_l = len(self.pos_paths)
        self.neg_l = len(self.neg_paths)

    def __getitem__(self, item):
        pos_img = Image.open(random.choice(self.pos_paths))
        neg_img = Image.open(random.choice(self.neg_paths))
        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return (pos_img, 0), (neg_img, 1)

    def __len__(self):
        return min(len(self.neg_paths), len(self.pos_paths))


_im_shape = [128, 128]

_transformer = tv.transforms.Compose([
    tv.transforms.Resize(_im_shape),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

_attr = 'Male'
_root_path = '../../Datasets/celebA/'
_dataset = celeba_dataset(_root_path, _attr, _transformer)

celeba_gender_loader = T.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=2)
