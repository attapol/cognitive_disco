import itertools

import numpy as np
import scipy as sp
import theano
import theano.sparse
import theano.tensor as T
import timeit

from learning import AdagradTrainer

class NeuralNet(object):
    """A wrapper neural net class to combine multiple layers into one network

    """

    def __init__(self, layers=None):
        self.params = []#a list of paramter variables
        self.input = [] #a list of input variables
        self.output = []#a list of output variables
        self.predict = [] # a list of prediction functions
        self.hinge_loss = None # a function
        self.crossentropy = None # a function
        self.layers = [] # a list of layer in the topological order
        self.rng = None
        if layers is not None:
            self.add_layers(layers)

    def add_layers(self, layers):
        self.rng = layers[0].rng 
        self.input = layers[0].input
        self.output = layers[-1].output
        self.activation = layers[-1].activation
        self.crossentropy = layers[-1].crossentropy
        self.hinge_loss = layers[-1].hinge_loss
        for layer in layers:
            self.layers.append(layer)
            self.params.extend(layer.params)

    def reset(self, rng):
        for layer in self.layers:
            layer.reset(rng)

class MJMModel(object):

    def __init__(self, layer_list, X_list):
        self.params = list(itertools.chain(*[layer.params for layer in layer_list]))
        self.input = [x for x in X_list]
        self.output = list(itertools.chain(*[layer.output for layer in layer_list]))
        self.predict = [layer.predict for layer in layer_list]

        #this can actually be anything, but we should play with weighting later
        #make the model focus on the main label sense
        self.crossentropy = 0
        self.hinge_loss = 0
        for layer in layer_list:
            self.crossentropy += layer.crossentropy
            self.hinge_loss += layer.hinge_loss

class GlueLayer(object):
    """Glue Layer that glues Bilinear and Linear layers output together
    by simply adding up
    """

    def __init__(self, layer_list, X_list, Y=None, activation_fn=T.tanh):
        net = 0
        for layer in layer_list:
            net += layer.activation
        self.activation = (
            net if activation_fn is None
            else activation_fn(net)
        )
        self.params = \
                list(itertools.chain(*[layer.params for layer in layer_list]))
        self.input = [x for x in X_list]
        if Y is None:
            self.output = []
        else:
            self.output = [Y]
            self.predict = [self.activation.argmax(1)]
            hinge_loss_instance, _ = theano.scan(
                    lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
                    sequences=[self.activation, Y])
            self.hinge_loss = hinge_loss_instance.sum()
            self.crossentropy = \
                    -T.mean(T.log(self.activation[T.arange(Y.shape[0]), Y]))

class MixtureOfExperts(object):

    def __init__(self, rng, n_in_list, expert_list, 
            X_list, Y, W_list=None, b=None):
        if W_list is None:
            W_list = []
            n_out = len(expert_list)
            for n_in in n_in_list:
                W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
                W = theano.shared(value=W_values, borrow=True)
                W_list.append(W)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.input = X_list    

        self.W_list = W_list
        self.b = b
        self.params = self.W_list + [self.b]
        self.expert_list = expert_list
        self.n_in_list = n_in_list
        for expert in expert_list:
            self.params.extend(expert.params)

        net = self.b
        for i in range(len(self.input)):
            if type(self.input[i]) == theano.sparse.basic.SparseVariable:
                net += theano.sparse.structured_dot(
                        self.input[i],self.W_list[i])
            else:
                net += T.dot(self.input[i], self.W_list[i])
        gating_activation = T.nnet.softmax(net)

        self.activation = 0
        for i, expert in enumerate(expert_list):
            g = T.addbroadcast(gating_activation[:,i:i+1], 1)
            self.activation += expert.activation * g

        self.output = [Y]
        self.predict = [self.activation.argmax(1)]
        hinge_loss_instance, _ = theano.scan(
                lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
                sequences=[self.activation, Y])
        self.hinge_loss = hinge_loss_instance.sum()
        self.crossentropy = -T.mean(T.log(self.activation[T.arange(Y.shape[0]), Y]))

    def reset(self, rng):
        for W, n_in in zip(self.W_list, self.n_in_list):
            total_n_in = np.sum(self.n_in_list)
            n_out = len(self.expert_list)
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (total_n_in + n_out)),
                    high=np.sqrt(6. / (total_n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W.set_value(W_values)
        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b.set_value(b_values)
        for expert in self.expert_list:
            expert.reset(rng)


class LinearLayer(object):
    """Linear Layer that supports multiple separate input (sparse) vectors

    """

    def __init__(self, rng, n_in_list, n_out, use_sparse, X_list=None, Y=None, 
            W_list=None, b=None, activation_fn=T.tanh):
        self.n_in_list = n_in_list
        self.n_out = n_out

        if W_list is None:
            W_list = []
            total_n_in = np.sum(n_in_list)
            for n_in in n_in_list:
                W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (total_n_in + n_out)),
                        high=np.sqrt(6. / (total_n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                W = theano.shared(value=W_values, borrow=True)
                W_list.append(W)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        if X_list is None:
            if use_sparse:
                self.input = [theano.sparse.csr_matrix() \
                        for i in xrange(len(n_in_list))]
            else:
                self.input = [T.matrix() for i in xrange(len(n_in_list))]
        else:
            assert(len(X_list) == len(n_in_list))
            self.input = X_list

        self.W_list = W_list
        self.b = b
        self.params = self.W_list + [self.b]
        self.activation_fn = activation_fn
        self.rng = rng
        
        net = self.b 
        for i in range(len(self.input)):
            if type(self.input[i]) == theano.sparse.basic.SparseVariable:
                net += theano.sparse.structured_dot(
                        self.input[i],self.W_list[i])
            else:
                net += T.dot(self.input[i], self.W_list[i])
        
        self.activation = (
            net if activation_fn is None
            else activation_fn(net)
        )

        if Y is None:
            self.output = []
            self.crossentropy = None
        else:
            self.output = [Y]
            self.predict = [self.activation.argmax(1)]
            hinge_loss_instance, _ = theano.scan(
                    lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
                    sequences=[self.activation, Y])
            self.hinge_loss = hinge_loss_instance.sum()
            self.crossentropy = \
                    -T.mean(T.log(self.activation[T.arange(Y.shape[0]), Y]))

    def reset(self, rng):
        for W, n_in in zip(self.W_list, self.n_in_list):
            total_n_in = np.sum(self.n_in_list)
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (total_n_in + self.n_out)),
                    high=np.sqrt(6. / (total_n_in + self.n_out)),
                    size=(n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
            W.set_value(W_values)
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b.set_value(b_values)

    def copy(self, X_list=None, use_sparse=False):
        l = LinearLayer(self.rng, self.n_in_list, self.n_out, use_sparse,
            X_list=X_list, W_list=self.W_list, b=self.b,
            Y=self.output[0] if len(self.output) != 0 else None, 
            activation_fn=self.activation_fn)
        return l

class LinearLayerTensorOutput(object):
    """Linear Layer that supports tensor-shaped output

    It turns out that I could have just modified the LinearLayer
    to compute crossentropy based on the boolean mask matrix. 
    But I will roll with this gratuitous class for now. 
    """

    def __init__(self, rng, n_in, n_out, X, W=None, b=None):
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), 
                    dtype=theano.config.floatX)
            W = theano.shared(value=W_values, borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.input = [X]
        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        net = X.dot(self.W) + self.b
        net_max = T.max(net, 2)
        denom = T.log(T.sum(T.exp(net - net_max[:,:,None]), 2)) - net_max 
        log_prob = net - denom[:,:,None]

        Y = T.tensor3(dtype=theano.config.floatX)
        self.output = [Y]
        num_nodes = T.sum(Y)
        self.crossentropy = -(log_prob * Y).sum() / num_nodes
        correct = (Y * T.eq(net, net_max[:,:,None])).sum()
        self.miscs = [correct / num_nodes, self.crossentropy]
        #node_mask = Y.sum(2).nonzero()
        #self.miscs = [T.argmax(Y,2)[node_mask][0:20], T.argmax(net, 2)[node_mask][0:20]]

    def reset(self, rng):
        n_in, n_out = self.W.get_value().shape
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),
            dtype=theano.config.floatX)
        self.W.set_value(W_values)
        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b.set_value(b_values)

class BilinearLayer(object):
    """Bilinear layer with dense vectors

    Tensor product in theano does not support sparse vectors. 
    We have to go around this isssue.
    """

    def __init__(self, rng, n_in1, n_in2, n_out, 
            X1=None, X2=None, Y=None, W=None, b=None, activation_fn=T.tanh):
        self.n_in1 = n_in1
        self.n_in2 = n_in2
        self.n_out = n_out
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in1 + n_in2 + n_out)),
                    high=np.sqrt(6. / (n_in1 + n_in2 + n_out)),
                    size=(n_out, n_in1, n_in2)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.X1 = T.matrix('x1') if X1 is None else X1
        self.X2 = T.matrix('x2') if X2 is None else X2
        self.W = W
        self.b = b
        self.input = [self.X1, self.X2]
        self.params = [self.W, self.b]
        net, _ = theano.scan(
                lambda x1, x2: T.dot(T.dot(x1, self.W),x2.T) + self.b,
                sequences=[self.X1, self.X2]
            )

        self.activation = net if activation_fn is None else activation_fn(net)
        if Y is None:
            self.output = []
        else:
            self.output = [Y]
            self.predict = [self.activation.argmax(1)]
            hinge_loss_instance, _ = theano.scan(
                    lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
                    sequences=[self.activation, Y])
            self.hinge_loss = hinge_loss_instance.sum()
            self.crossentropy = \
                    -T.mean(T.log(self.activation[T.arange(Y.shape[0]), Y]))

    def reset(self, rng):
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (self.n_in1 + self.n_in2 + self.n_out)),
                high=np.sqrt(6. / (self.n_in1 + self.n_in2 + self.n_out)),
                size=(self.n_out, self.n_in1, self.n_in2)
            ),
            dtype=theano.config.floatX
        )
        self.W.set_value(W_values)
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b.set_value(b_values)


def test_bilinear():
    num_features = 50
    n_out = 3
    num = 2000
    x1 = np.random.randn(num, num_features)
    x2 = np.random.randn(num, num_features)

    s = np.random.randn(num, n_out)
    y = s.argmax(1)

    rng = np.random.RandomState(12)

    blm = BilinearLayer(rng, num_features, num_features, n_out, activation_fn=T.tanh)
    trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
    trainer.train([x1,x2,y], 10)


def test_linear_layers(num_features1, num_feature2, n_out, num):

    #x1 = sp.sparse.rand(num, num_features1).todense()
    x1 = np.random.randn(num, num_features1)
    w1 = np.random.randn(num_features1, n_out)
    s1 = x1.dot(w1)

    #x2 = sp.sparse.rand(num, num_features2).todense()
    x2 = np.random.randn(num, num_features2)
    w2 = np.random.randn(num_features2, n_out)
    s2 = x2.dot(w2)

    y = (s1 + s2).argmax(1)

    train = [x1[0:num/2,:], x2[0:num/2,:], y[0:num/2]]
    dev = [x1[num/2:,:], x2[num/2:,:], y[num/2:]]

    rng = np.random.RandomState(12)

    lm = LinearLayer(rng, [num_features1, num_features2], n_out, False,
            Y=T.lvector(), activation_fn=None)
    print 'Training with hinge loss'
    trainer = AdagradTrainer(lm, lm.hinge_loss, 0.01, 0.01)
    start_time = timeit.default_timer()
    trainer.train_minibatch(100, 20, train, dev, dev)
    end_time = timeit.default_timer()
    print end_time - start_time 
    
def test_sparse_linear_layers(num_features1, num_features2, n_out, num):

    x1 = sp.sparse.rand(num, num_features1).tocsr()
    w1 = np.random.randn(num_features1, n_out)
    s1 = x1.dot(w1)

    x2 = sp.sparse.rand(num, num_features2).tocsr()
    w2 = np.random.randn(num_features2, n_out)
    s2 = x2.dot(w2)

    y = (s1 + s2).argmax(1)

    train = [x1[0:num/2,:], x2[0:num/2,:], y[0:num/2]]
    dev = [x1[num/2:,:], x2[num/2:,:], y[num/2:]]

    from learning import AdagradTrainer
    rng = np.random.RandomState(12)

    X1 = theano.sparse.csr_matrix('x1')
    X2 = theano.sparse.csr_matrix('x2')

    lm = LinearLayer(rng, [num_features1, num_features2], n_out, True, X_list=[X1, X2],
            Y=T.lvector(), activation_fn=None)
    print 'Training with hinge loss'
    trainer = AdagradTrainer(lm, lm.hinge_loss, 0.01, 0.01)
    start_time = timeit.default_timer()
    trainer.train_minibatch(100, 20, train, dev, dev)
    end_time = timeit.default_timer()
    print end_time - start_time 


if __name__ == '__main__':
    num_features1 = 4000
    num_features2 = 2000
    n_out = 3
    num = 10000
    test_sparse_linear_layers(num_features1, num_features2, n_out, num)
    test_linear_layers(num_features1, num_features2, n_out, num)
