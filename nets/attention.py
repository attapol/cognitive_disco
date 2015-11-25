import theano
import theano.tensor as T
from theano import config
import numpy as np

class AttentionModelBase(object):

    def setup(self):
        self._init_params()
        self.X = T.tensor3('x', dtype=config.floatX) 
        self.mask = T.matrix('mask', dtype=config.floatX)
        self.input = [self.X, self.mask]
        self.a = self._compute_a()
        net = self.X * self.a[:,:,None] * self.mask[:,:,None]
        self.activation = net.sum(axis=0)

class AttentionModelSimple(AttentionModelBase):

    def __init__(self, rng, num_units):
        self.rng = rng
        self.num_units = num_units
        self.setup()

    def _compute_a(self):
        return 1 / (1+ T.exp(-(self.X.dot(self.W) + self.b[0])))

    def _get_init_param_values(self):
        W_values = np.zeros(self.num_units, dtype=config.floatX)
        b_value = np.zeros(1, dtype=config.floatX)
        return W_values, b_value

    def _init_params(self):
        W_values, b_value = self._get_init_param_values()
        self.W = theano.shared(W_values, borrow=True)
        self.b = theano.shared(b_value, borrow=True)
        self.params = [self.W, self.b]

    def reset(self, rng):
        W_values = np.zeros(self.num_units, dtype=config.floatX)
        b_value = np.zeros(1, dtype=config.floatX)
        self.W.set_value(W_values)
        self.b.set_value(b_value)

class AttentionModelSimpleHiddenLayer(AttentionModelBase):

    def __init__(self, rng, num_units, num_hidden_units):
        self.rng = rng
        self.num_units = num_units
        self.num_hidden_units = num_hidden_units
        self.setup()

    def _compute_a(self):
        hidden = T.tanh(self.X.dot(self.H) + self.bH)
        return 1 / (1+ T.exp(-(hidden.dot(self.W) + self.b[0])))

    def _get_init_param_values(self):
        H_values = np.zeros((self.num_units, self.num_hidden_units),
                dtype=config.floatX)
        bH_values = np.zeros(self.num_hidden_units, dtype=config.floatX)
        W_values = np.zeros(self.num_hidden_units, dtype=config.floatX)
        b_value = np.zeros(1, dtype=config.floatX)
        return H_values, bH_values, W_values, b_value

    def _init_params(self):
        H_values, bH_values, W_values, b_value = \
                self._get_init_param_values()
        self.H = theano.shared(H_values, borrow=True)
        self.bH = theano.shared(bH_values, borrow=True)
        self.W = theano.shared(W_values, borrow=True)
        self.b = theano.shared(b_value, borrow=True)
        self.params = [self.W, self.b, self.H, self.bH]

    def reset(self, rng):
        H_values, bH_values, W_values, b_value = \
                self._get_init_param_values()
        self.H.set_value(H_values)
        self.bH.set_value(bH_values)
        self.W.set_value(W_values)
        self.b.set_value(b_value)
