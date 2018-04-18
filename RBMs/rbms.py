from base import rbm_base
import torch as T

softplus = lambda x: T.log(1 + T.exp(x))


class BernoulliRBM(rbm_base):
    """RBM with Bernoulli both visible and hidden units."""

    def __init__(self, model_path='./logs/brbm/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(model_path=model_path, *args, **kwargs)

    def _h_given_v(self, v):
        h_probs = T.sigmoid(v @ self._W + self._hb)
        h_samples = T.bernoulli(h_probs)
        return h_probs, h_samples

    def _v_given_h(self, h):
        v_probs = T.sigmoid(h @ self._W.t() + self._vb)
        v_samples = T.bernoulli(v_probs)
        return v_probs, v_samples

    def _free_energy(self, v):
        h = v @ self._W + self._hb
        t1 = - v @ self._vb
        t2 = - T.sum(softplus(h), 1)
        return T.mean(t1 + t2)
