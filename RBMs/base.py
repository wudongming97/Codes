from utils import *
import os
from tensorboardX import SummaryWriter


class rbm_base:
    def __init__(self, model_path, drop_probs=0.0, n_epoch_to_save=2,
                 v_sz=784, v_layer_cls=None, v_layer_params=None,
                 h_sz=256, h_layer_cls=None, h_layer_params=None, pcd=True,
                 W_init=None, vb_init=None, hb_init=None, metrics_interval=200, verbose=True,
                 epoch_start_decay=1, epoch_stop_decay=8, ultimate_lr=2e-5,
                 n_gibbs_steps=1, sample_v_states=True, sample_h_states=True,
                 lr=1e-2, momentum=0.9, max_epoch=10, batch_size=16, l2=1e-4):

        self.model_path = model_path
        self.drop_probs = drop_probs
        self.pcd = pcd
        self.persistent_chains = None
        self.n_epoch_to_save = n_epoch_to_save
        self._writer = SummaryWriter(model_path)
        self.v_sz = v_sz
        self.h_sz = h_sz

        v_layer_params = v_layer_params or {}
        v_layer_params.setdefault('n_units', self.v_sz)
        self._v_layer = v_layer_cls(**v_layer_params)

        h_layer_params = h_layer_params or {}
        h_layer_params.setdefault('n_units', self.h_sz)
        self._h_layer = h_layer_cls(**h_layer_params)

        self.W_init = W_init
        self.vb_init = vb_init
        self.hb_init = hb_init

        self.metrics_interval = metrics_interval
        self.verbose = verbose

        self.epoch_start_decay = epoch_start_decay
        self.epoch_stop_decay = epoch_stop_decay
        self.ultimate_lr = ultimate_lr

        self.sample_v_states = sample_v_states
        self.sample_h_states = sample_h_states
        self.n_gibbs_steps = n_gibbs_steps

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.l2 = l2

        # 带下划线的成员，是变化的，是save要保存的变量
        self._step = 0
        self._lr = lr
        self._momentum = momentum
        self._dW = Tensor(np.zeros([self.v_sz, self.h_sz]))
        self._dhb = Tensor(np.zeros(self.h_sz))
        self._dvb = Tensor(np.zeros(self.v_sz))

        self._W = Tensor(np.random.uniform(size=(self.v_sz, self.h_sz))) if self.W_init is None else self.W_init
        self._hb = Tensor(np.zeros(self.h_sz)) if self.hb_init is None else self.hb_init
        self._vb = Tensor(np.zeros(self.v_sz)) if self.vb_init is None else self.vb_init

    def _dropout(self, X, drop_probs):
        assert 0 <= drop_probs < 1
        X *= T.bernoulli((1 - drop_probs) * T.ones_like(X)) / (1 - drop_probs)
        return X

    def _free_energy(self, v):
        raise NotImplementedError

    def _free_energy_gap_metric(self, train, valid, batch_size):
        train_feg = self._free_energy(shuffle_batch(train, batch_size).view(-1, self.v_sz))
        valid_feg = self._free_energy(shuffle_batch(valid, batch_size).view(-1, self.v_sz))
        return train_feg - valid_feg

    def _weight_decay(self, epoch_cur, epoch_start, epoch_stop, ultimate_lr):
        assert epoch_start < epoch_stop
        if epoch_cur > epoch_start:
            epoch_stop = self.max_epoch if epoch_stop > self.max_epoch else epoch_stop
            self._lr -= (self._lr - ultimate_lr) * (epoch_cur - epoch_start) / (epoch_stop - epoch_start)

    def _h_given_v(self, v):
        h_probs = self._h_layer.activation(v @ self._W + self._hb)
        h_samples = self._h_layer.sample(h_probs)
        return h_probs, h_samples

    def _v_given_h(self, h):
        v_probs = self._v_layer.activation(h @ self._W.t() + self._vb)
        v_samples = self._v_layer.sample(v_probs)
        return v_probs, v_samples

    def _gibbs_step(self, h0):
        v_probs, v_samples = self._v_given_h(h0)
        v = v_samples if self.sample_v_states else v_probs
        h_probs, h_samples = self._h_given_v(v)
        h = h_samples if self.sample_h_states else h_probs
        return v, h

    def _gibbs_chain(self, h0, n_gibbs_steps):
        for _ in range(n_gibbs_steps):
            v, h = self._gibbs_step(h0)
        return v, h

    def _update(self, v0):
        N = v0.size()[0]

        h0 = self._h_given_v(v0)[0]
        h_gibbs = self.persistent_chains if self.pcd else h0
        vn, hn = self._gibbs_chain(h_gibbs, self.n_gibbs_steps)
        self.persistent_chains = hn

        dW = (vn.t() @ hn - v0.t() @ h0) / N - self.l2 * self._W
        dvb = T.mean(vn - v0, 0)
        dhb = T.mean(hn - h0, 0)

        # todo 添加稀疏化正则项

        # update
        self._dW = self._lr * (self._momentum * self._dW + dW)
        self._dvb = self._lr * (self._momentum * self._dvb + dvb)
        self._dhb = self._lr * (self._momentum * self._dhb + dhb)
        self._W = self._W - self._dW
        self._vb = self._vb - self._dvb
        self._hb = self._hb - self._dhb

    def fit(self, X, X_val):
        self.persistent_chains = self._h_given_v(
            shuffle_batch(X, self.batch_size).view(self.batch_size, self.v_sz))[0]
        for epoch in range(self.max_epoch):
            self._weight_decay(epoch, self.epoch_start_decay, self.epoch_stop_decay, self.ultimate_lr)
            for X_batch in next_batch(X, self.batch_size):
                X_batch = X_batch.view(-1, self.v_sz)
                X_batch = self._dropout(X_batch, self.drop_probs)
                self._update(X_batch)
                self._step += 1

                # verbose and metrics
                if (self._step + 1) % self.metrics_interval == 0:
                    gap = self._free_energy_gap_metric(X, X_val, 200)
                    self._writer.add_scalar('free_energy_gap', gap, self._step)
                    self._writer.add_scalar('lr', self._lr, self._step)

                    if self.verbose:
                        print('epoch: [%d \ %d] global_step: [%d] free_energy_gap: [%.3f]' % (
                            epoch, self.max_epoch, self._step, gap))
            # save
            if (epoch + 1) % self.n_epoch_to_save:
                self.save()

    def _inf(self, h0, n_gibbs_steps, to_numpy=True):
        v, h = self._gibbs_chain(h0, n_gibbs_steps)
        if to_numpy:
            return v.cpu().numpy(), h.cpu().numpy()
        else:
            return v, h

    def inf_from_valid(self, batch_X_val, n_gibbs_steps):
        batch_X_val = batch_X_val.view([-1, self.v_sz])
        h0_val, _ = self._h_given_v(batch_X_val)
        v_inf, _ = self._inf(h0_val, n_gibbs_steps, to_numpy=True)
        return v_inf

    def inf_by_stochastic(self, batch_size, n_gibbs_steps):
        h0_sto = self._h_layer.init(batch_size)
        v_inf, _ = self._inf(h0_sto, n_gibbs_steps, to_numpy=True)
        return v_inf

    def save(self):
        np.savez(os.path.join(self.model_path + 'ckpt_{}.npz'.format(self._step)),
                 W=self._W, vb=self._vb, hb=self._hb, dW=self._dW, dvb=self._dvb, dhb=self._dhb,
                 optim_params=np.array([self._lr, self._momentum, self._step]))
        from shutil import copyfile
        copyfile(os.path.join(self.model_path + 'ckpt_{}.npz'.format(self._step)),
                 os.path.join(self.model_path + 'ckpt_latest.npz'))

    def load(self, step_to_load=None, only_weights=False):
        if step_to_load is None:
            npz_file = os.path.join(self.model_path + 'ckpt_latest.npz')
        else:
            npz_file = os.path.join(self.model_path + 'ckpt_{}.npz'.format(step_to_load))

        if not os.path.isfile(npz_file):
            return False

        npz = np.load(npz_file)
        self._W = Tensor(npz['W'])
        self._vb = Tensor(npz['vb'])
        self._hb = Tensor(npz['hb'])
        if not only_weights:
            self._lr, self._momentum, self._step = npz['optim_params']
            self._dW = Tensor(npz['dW'])
            self._dhb = Tensor(npz['dhb'])
            self._dvb = Tensor(npz['dvb'])
        return True
