import theano
import theano.tensor as T
from theano import config
import numpy as np

from bilinear_layer import make_multilayer_net

class AttentionModelBase(object):

    def compute_activations(self):
        self.a_train, self.a_test = self._compute_attention()

        net_train = self.X * self.a_train[:,:,None] * self.mask[:,:,None]
        net_test = self.X * self.a_test[:,:,None] * self.mask[:,:,None]
        self.activation_train = net_train.sum(axis=0)
        self.activation_test = net_test.sum(axis=0)

class AttentionModelSimple(AttentionModelBase):

    def __init__(self, rng, num_units, num_hidden_layers,
            num_hidden_units, dropout):
        self.rng = rng
        self.num_units = num_units
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.n_out = num_units

        self.X = T.tensor3('x', dtype=config.floatX) 
        self.mask = T.matrix('mask', dtype=config.floatX)
        self.input = [self.X, self.mask]
        
        self.att_net, _ = make_multilayer_net(rng, 
                num_units, self.X, None, False, 
                num_hidden_layers, num_hidden_units, 1, 
                lambda x: 1 / (1 + T.exp(-x)), dropout)
        self.params = self.att_net.params
        self.compute_activations()

    def _compute_attention(self):
        a_train = self.att_net.activation_train
        a_test = self.att_net.activation_test
        return a_train, a_test

    def reset(self, rng):
        self.att_net.reset(rng)
