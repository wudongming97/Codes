from utils import *


class BaseLayer:
    """Class encapsulating one layer of stochastic units."""

    def __init__(self, n_units, *args, **kwargs):
        super(BaseLayer, self).__init__()
        self.n_units = n_units

    def init(self, bz):
        """Randomly initialize states according to their distribution."""
        raise NotImplementedError

    def activation(self, x):
        raise NotImplementedError

    def sample(self, probs):
        raise NotImplementedError


class BernoulliLayer(BaseLayer):
    def __init__(self, *args, **kwargs):
        super(BernoulliLayer, self).__init__(*args, **kwargs)

    def init(self, bz):
        return Tensor(np.zeros([bz, self.n_units]))

    def activation(self, x):
        return T.sigmoid(x)

    def sample(self, probs):
        return T.bernoulli(probs)
