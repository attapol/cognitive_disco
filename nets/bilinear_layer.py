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

			true_label_activation, _ = theano.scan(
					lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
					sequences=[self.activation, self.Y])
			self.hinge_loss = true_label_activation.sum()

			#true_label_activation, _ = theano.scan(
					#lambda a, y: T.log(1 - a[y]), 
					#sequences=[self.activation, self.Y])
			#self.crossentropy = true_label_activation.mean()
			self.crossentropy = -T.mean(self.activation[T.arange(self.Y.shape[0]), self.Y])

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

		true_label_activation, _ = theano.scan(
				lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
				sequences=[self.activation, self.Y])
		self.hinge_loss = true_label_activation.sum()

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

if __name__ == '__main__':
	test_hidden()
