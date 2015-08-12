import numpy as np
import theano
from theano import sparse
import theano.sparse
import theano.tensor as T

class Layer(object):

	def __init__(self, rng, n_in, n_out, use_sparse, X=None, Y=None,
			W=None, b=None, activation_fn=T.tanh):
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			W = theano.shared(value=W_values, borrow=True)
		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, borrow=True)

		if X is None:
			X = T.matrix()
			self.input = [X]
		self.X = X

		self.W = W
		self.b = b
		self.params = [self.W, self.b]
		
		net = T.dot(X, W) + b

		self.activation = (
			net if activation_fn is None
			else activation_fn(net)
		)
		self.Y = Y
		if self.Y is not None:
			self.output = [self.Y]
			self.predict = self.activation.argmax(1)

			hinge_loss_instance, _ = theano.scan(
					lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
					sequences=[self.activation, self.Y])
			self.hinge_loss = hinge_loss_instance.sum()
			self.crossentropy = -T.mean(self.activation[T.arange(self.Y.shape[0]), self.Y])

class LinearLayer(object):
	"""Linear Layer that supports multiple separate input (sparse) vectors

	"""

	def __init__(self, rng, n_in_list, n_out, X_list=None, Y=None, 
			W_list=None, b=None, activation_fn=T.tanh):
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
			self.input = [T.matrix() for i in xrange(len(n_in_list))]
		else:
			assert(len(X_list) == len(n_in_list))
			self.input = X_list

		self.W_list = W_list
		self.b = b
		self.params = self.W_list + [self.b]
		
		net = self.b + T.dot(self.input[0], self.W_list[0])
		for i in range(1, len(self.input)):
			net += T.dot(self.input[i], self.W_list[i])
		
		self.activation = (
			net if activation_fn is None
			else activation_fn(net)
		)

		if Y is None:
			self.output = [self.activation]
		else:
			self.output = [Y]
			self.predict = self.activation.argmax(1)
			hinge_loss_instance, _ = theano.scan(
					lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
					sequences=[self.activation, Y])
			self.hinge_loss = hinge_loss_instance.sum()
			self.crossentropy = -T.mean(self.activation[T.arange(Y.shape[0]), Y])


class BilinearLayer(object):

	def __init__(self, rng, n_in1, n_in2, n_out, W=None, b=None, activation_fn=T.tanh):
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
		
		self.X1 = T.matrix('x1')
		self.X2 = T.matrix('x2')
		self.Y = T.lvector('y')
		self.W = W
		self.b = b
		self.input = [self.X1, self.X2]
		self.output = [self.Y]
		self.params = [self.W, self.b]

		#if use_sparse:
		#	net, _ = theano.scan(
		#			lambda i, x1, x2: sparse.dot(sparse.dot(x1[i:i+1], self.W),x2[i:i+1].T) + self.b,
					#sequences=T.arange(10),
		#			sequences=[T.arange(self.X1.shape[0])],
		#			non_sequences = [self.X1, self.X2],
		#		)
		#else:
		net, _ = theano.scan(
				lambda x1, x2: T.dot(T.dot(x1, self.W),x2.T) + self.b,
				sequences=[self.X1, self.X2]
			)


		self.activation = (
			net if activation_fn is None
			else activation_fn(net)
		)
		self.predict = self.activation.argmax(1)

		hinge_loss_instance, _ = theano.scan(
				lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
				sequences=[self.activation, self.Y])
		self.hinge_loss = hinge_loss_instance.sum()

		true_label_activation, _ = theano.scan(
				lambda a, y: T.log(1 - a[y]), 
				sequences=[self.activation, self.Y])
		self.crossentropy = true_label_activation.mean()

def test_bilinear():
	num_features = 50
	n_out = 3
	num = 2000
	x1 = np.random.randn(num, num_features)
	x2 = np.random.randn(num, num_features)

	s = np.random.randn(num, n_out)
	y = s.argmax(1)

	rng = np.random.RandomState(12)

	from learning import AdagradTrainer
	blm = BilinearLayer(rng, num_features, num_features, n_out, activation_fn=T.tanh)
	trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
	#W = theano.shared(value=np.zeros((n_out,num_features,num_features)), name='W', borrow=True)
	#blm = BilinearLayer(rng, num_features, num_features, n_out, W=W, activation_fn=T.nnet.softmax)
	#trainer = AdagradTrainer(blm, blm.crossentropy(), 0.01, 0.01)
	trainer.train([x1,x2,y], 10)

def test_hidden():
	num_features = 100
	n_out = 3
	num = 10000
	x = np.random.randn(num, num_features)
	#s = np.random.randn(num, n_out)
	w = np.random.randn(num_features, n_out)
	s = x.dot(w)
	y = s.argmax(1)

	train = [x[0:num/2,:], y[0:num/2]]
	dev = [x[num/2:,:], y[num/2:]]

	from learning import AdagradTrainer
	rng = np.random.RandomState(12)

	lm = Layer(rng, num_features, n_out, False, Y=T.lvector(), activation_fn=None)
	print 'Training with hinge loss'
	trainer = AdagradTrainer(lm, lm.hinge_loss, 0.01, 0.01)
	trainer.train_minibatch(100, 100, train, dev, dev)

	print 'Training with crossentropy'
	lm = Layer(rng, num_features, n_out, False, Y=T.lvector(), activation_fn=T.nnet.softmax)
	trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
	trainer.train_minibatch(100, 100, train, dev, dev)

def test_linear_layers():
	num_features1 = 100
	num_features2 = 50
	n_out = 3
	num = 10000

	x1 = np.random.randn(num, num_features1)
	w1 = np.random.randn(num_features1, n_out)
	s1 = x1.dot(w1)

	x2 = np.random.randn(num, num_features2)
	w2 = np.random.randn(num_features2, n_out)
	s2 = x2.dot(w2)

	y = (s1 + s2).argmax(1)

	train = [x1[0:num/2,:], x2[0:num/2,:], y[0:num/2]]
	dev = [x1[num/2:,:], x2[num/2:,:], y[num/2:]]

	from learning import AdagradTrainer
	rng = np.random.RandomState(12)

	lm = LinearLayer(rng, [num_features1, num_features2], n_out, 
			Y=T.lvector(), activation_fn=None)
	print 'Training with hinge loss'
	trainer = AdagradTrainer(lm, lm.hinge_loss, 0.01, 0.01)
	trainer.train_minibatch(100, 100, train, dev, dev)

def test_sparse_linear_layers():


if __name__ == '__main__':
	test_linear_layers()
