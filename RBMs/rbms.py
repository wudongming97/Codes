from base import rbm_base
from layers import *

softplus = lambda x: T.log(1 + T.exp(x))


class BernoulliRBM(rbm_base):
    """RBM with Bernoulli both visible and hidden units."""

    def __init__(self, model_path='./logs/brbm/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                           h_layer_cls=BernoulliLayer,
                                           model_path=model_path, *args, **kwargs)

    def _free_energy(self, v):
        h = v @ self._W + self._hb
        t1 = - v @ self._vb
        t2 = - T.sum(softplus(h), 1)
        return T.mean(t1 + t2)
