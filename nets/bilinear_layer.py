import numpy as np
import theano
import theano.tensor as T

class BilinearLayer(object):

	def __init__(self, rng, input1, input2, n_in1, n_in2, n_out, 
			W=None, b=None, activation=T.tanh):
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
		
		self.input1 = input1
		self.input2 = input2
		self.W = W
		self.b = b

		self.params = [self.W, self.b]
		raw_output, _ = theano.scan(
				lambda x1, x2: T.dot(T.dot(x1, self.W),x2.T) + self.b,
				sequences=[self.input1, self.input2]
			)
		self.output = (
			raw_output if activation is None
			else activation(raw_output)
		)
		self.predict = self.output.argmax(1)

	def cost(self, y):
		return T.maximum(0, 1 - self.output[y] + self.output).sum() - 1

if __name__ == '__main__':
	num_features = 5000
	n_out = 3
	num = 20
	x1 = np.random.randn(num, num_features)
	x2 = np.random.randn(num, num_features)

	s = np.random.randn(num, n_out)
	y = s.argmax(1)

	X1 = T.matrix('x1')
	X2 = T.matrix('x2')
	rng = np.random.RandomState(12)
	blm = BilinearLayer(rng, X1, X2, num_features, num_features, n_out)

	o = theano.function(
			inputs=[X1, X2], 
			outputs = [blm.output]
			)
	c = blm.cost(y)
	gparams = [T.grad(cost=c, wrt=x) for x in blm.params]
	updates = [ (param, param - 0.001 * gparam)
        for param, gparam in zip(blm.params, gparams)
    ]
	train = theano.function(
			inputs=[X1, X2], 
			outputs = [c],
			updates= updates)
	for i in range(10):
		print train(x1,x2)

