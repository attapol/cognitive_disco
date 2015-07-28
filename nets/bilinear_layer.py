import numpy as np
import theano
import theano.tensor as T

class BilinearLayer(object):

	def __init__(self, rng, n_in1, n_in2, n_out, 
			W=None, b=None, activation_fn=T.tanh):
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in1 + n_in2 + n_out)),
					high=np.sqrt(6. / (n_in1 + n_in2 + n_out)),
					#size=(n_in1, n_in2)
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

		raw_output, _ = theano.scan(
				lambda x1, x2: T.dot(T.dot(x1, self.W),x2.T) + self.b,
				sequences=[self.X1, self.X2]
			)

		self.activation = (
			raw_output if activation_fn is None
			else activation_fn(raw_output)
		)
		self.predict = self.activation.argmax(1)

	def hinge_loss(self):
		true_label_activation, _ = theano.scan(
				lambda a, y: T.maximum(0, 1 - a[y] + a).sum() - 1 ,
				sequences=[self.activation, self.Y])
		return true_label_activation.sum()

	def crossentropy(self):
		true_label_activation, _ = theano.scan(
				lambda a, y: T.log(1 - a[y]), 
				sequences=[self.activation, self.Y])
		return true_label_activation.mean()

if __name__ == '__main__':
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
	trainer = AdagradTrainer(blm, blm.hinge_loss(), 0.01, 0.01)
	#W = theano.shared(value=np.zeros((n_out,num_features,num_features)), name='W', borrow=True)
	#blm = BilinearLayer(rng, num_features, num_features, n_out, W=W, activation_fn=T.nnet.softmax)
	#trainer = AdagradTrainer(blm, blm.crossentropy(), 0.01, 0.01)
	trainer.train([x1,x2,y], 10)
