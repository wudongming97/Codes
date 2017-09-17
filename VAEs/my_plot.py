import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



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