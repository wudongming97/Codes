import math
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

Batch_sz = 30
MNIST_train_loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST('../datasets/mnist', train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=Batch_sz, shuffle=True)

#normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
#transform = transforms.Compose([transforms.ToTensor(),
#                                    transforms.Normalize((0.1307,), (0.3081,))])
#train_datasets = fashion(root='../datasets/fashion', train=True, transform=transform, download=True)


def my_plot(save_dir, filename, samples, n):
    width = math.ceil(math.sqrt(n))
    sz = int(math.sqrt(samples.shape[1]))
    fig = plt.figure(figsize=(width, width))
    gs = gridspec.GridSpec(width, width)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        if i >= n:
            break
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(sz, sz), cmap='Greys_r')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '{}.png'.format(filename, bbox_inches='tight'))
    plt.close(fig)