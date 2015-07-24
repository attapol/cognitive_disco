import numpy
import theano
import theano.tensor as T

class MaxEnt(object):

	def __init__(self, n_in, n_out, W=None, b=None):
		if W is None:
			W_values = numpy.zeros((n_in, n_out))
			W = theano.shared(value=W_values, name='W', borrow=True)
		
		if b is None:
			b_values = numpy.zeros((n_out))
			b = theano.shared(value=b_values, name='B', borrow=True)

		self.x = T.matrix('x')
		self.y = T.lvector('y')
		self.W = W
		self.b = b

		self.params = [self.W, self.b]
		self.output = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
		self.predict = self.output.argmax(1)
		self.cost = -T.mean(T.log(self.output[T.arange(self.y.shape[0]), self.y]))
		self.gW, self.gb = T.grad(cost=self.cost, wrt=[self.W, self.b])


	def train(self, training_data):
		updates = [(self.W, self.W - 0.1 * self.gW), (self.b, self.b - 0.1 * self.gb)]
		index = T.lscalar()
		batch_size = 20
		train_model = theano.function(
				inputs=[index], 
				outputs = [self.output, self.cost],
				updates = updates,
				givens={
					self.x : training_data[0][index*batch_size:(index+1)*batch_size, :],
					self.y : training_data[1][index*batch_size:(index+1)*batch_size]
				}
				)
		for t in range(100):
			for i in range(100):
				o,c= train_model(i)
				print c 


	def train_batch(self, training_data):
		updates = [(self.W, self.W - 0.001 * self.gW), (self.b, self.b - 0.001 * self.gb)]
		index = T.lscalar()
		train_model = theano.function(
				inputs= [self.x, self.y],
				outputs = [self.output, self.cost],
				updates = updates,
				)
		for t in range(100):
			o,c= train_model(training_data[0].get_value(), training_data[1].get_value())
			print c

	def test_accuracy(self, data):
		classify = theano.function(
				inputs = [self.x],
				outputs = [self.predict],
				)
		predictions = classify(data[0].get_value())
		print (predictions == data[1].get_value()).sum() / float(data[1].get_value().size)

if __name__ == '__main__':
	n_feats = 500
	n_out = 3
	n = 50000
	m = MaxEnt(n_feats, n_out)

	y_scores = numpy.random.randn(n, n_out)
	data_y = theano.shared(y_scores.argmax(1))

	x = numpy.random.randn(n, n_feats)
	data_x = theano.shared(x)
	training_data = (data_x, data_y)
	m.train_batch(training_data)

	y_scores = numpy.random.randn(n, n_out)
	data_y = theano.shared(y_scores.argmax(1))

	x = numpy.random.randn(n, n_feats)
	data_x = theano.shared(x)
	test_data = (data_x, data_y)
	m.test_accuracy(test_data)
