import os.path

from utils import *

Tensor = torch.cuda.FloatTensor if USE_GPU else torch.FloatTensor
torch.manual_seed(1234)


class rbm:
    def __init__(self, v_sz, h_sz, lr, k, logdir):
        self.v_sz = v_sz
        self.h_sz = h_sz
        self.k = k
        self.lr = lr
        self.logdir = logdir
        self.w = Tensor(np.random.normal(1e-7, 2e-2, size=[self.v_sz, self.h_sz]))
        self.hb = Tensor(np.zeros(self.h_sz))

        self.vb = Tensor(np.zeros(self.v_sz))

    def fit(self, datas, epoch):
        start_to_decay = epoch // 2
        for e in range(epoch):
            for i, (images, _) in enumerate(datas):
                images = images.view([-1, self.v_sz])
                images = images.cuda() if USE_GPU else None
                self._steps(Tensor(images))
                if i % 100 == 0:
                    loss = self._metric(images)
                    print('Epochs [{}/{}] Batchs [{}] loss [{}]'.format(e, epoch, i, loss))

            if e > start_to_decay:  # 在start_to_decay后， lr线性减少
                self.lr = self.lr - self.lr * (e - start_to_decay) / (epoch - start_to_decay)

            self.save('checkpoint_e_{}.npz'.format(e))
            self.save('checkpoint_latest.npz')

    def inference(self, sz, k):
        v0 = Tensor(np.random.uniform(0, 1, [sz, self.v_sz]))
        _, _, pv, v, _, _ = self._gibbs(v0, k)
        return pv, v

    def save(self, file):
        np.savez(self.logdir + file, w=self.w, vb=self.vb, hb=self.hb)

    def load(self, file='checkpoint_latest.npz'):
        npz_file = self.logdir + file
        if not os.path.isfile(npz_file):
            return False
        else:
            npz = np.load(npz_file)
            self.w = Tensor(npz['w'])
            self.vb = Tensor(npz['vb'])
            self.hb = Tensor(npz['hb'])
            return True

    def _ph_v(self, v):
        vwb = v @ self.w + self.hb
        return torch.sigmoid(vwb)

    def _pv_h(self, h):
        hwb = h @ self.w.t() + self.vb
        return torch.sigmoid(hwb)

    def _gibbs(self, v0, k):
        ph0 = self._ph_v(v0)
        pv, v, ph, h = None, v0, ph0, torch.bernoulli(ph0)
        for _ in range(k):
            pv = self._pv_h(h)
            v = torch.bernoulli(pv)
            ph = self._ph_v(v)
            h = torch.bernoulli(ph)

        return v0, ph0, pv, v, ph, h

    def _steps(self, v):
        bz = v.size()[0]

        v0, ph0, _, v_n, ph_n, _ = self._gibbs(v, self.k)
        self.w = self.w - self.lr * (v_n.t() @ ph_n - v0.t() @ ph0) / bz
        self.hb = self.hb - self.lr * torch.mean(ph_n - ph0, 0)
        self.vb = self.vb - self.lr * torch.mean(v_n - v0, 0)

    def _metric(self, v0, method='bce'):
        _, _, pv, _, _, _ = self._gibbs(v0, self.k)
        if method == 'bce':  # cd算法用bce更好
            return torch.mean((pv - v0) ** 2)
        raise ValueError


if __name__ == '__main__':
    rbm = rbm(784, 1024, 1e-3, 10, './results/rbm/')
    if not rbm.load():
        rbm.fit(mnist, 10)

    # 随机生成
    pv, v = rbm.inference(100, 200)
    for t, im in enumerate([pv, v]):
        images = np.reshape(im.cpu().numpy(), [-1, 28, 28, 1])
        imsave_('./results/rbm/mnist_{}.png'.format(t), imcombind_(images))
