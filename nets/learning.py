import numpy as np
import theano
import theano.tensor as T

class AdagradTrainer(object):

	def __init__(self, model, cost_function, learning_rate, lr_smoother):
		self.model = model
		self.cost_function = cost_function 
		self.learning_rate = learning_rate
		self.lr_smoother = lr_smoother

		gparams = [T.grad(cost=cost_function, wrt=x) for x in self.model.params]
		adagrad_rates = [T.shared(value=np.zeros(param.get_value().shape)) 
				for param in self.model.params]
		sum_gradient_squareds = [T.shared(value=np.zeros(param.get_value().shape)) 
				for param in self.model.params]

		self.sgs_updates = [(sgs, sgs + T.square(gparam)) 
			for sgs, gparam in zip(sum_gradient_squareds, gparams)]

		self.adagrad_lr_updates = [(adagrad_rate, 
			adagrad_rate + learning_rate / (lr_smoother + T.sqrt(sum_gradient_squared)))
			for adagrad_rate, sum_gradient_squared in zip(adagrad_rates, sum_gradient_squareds)]

		self.param_updates = [(param, param - adagrad_rate * gparam) 
				for param, gparam, adagrad_rate in zip(self.model.params, gparams, adagrad_rates)]

	def train(self, training_data, mini_batch_size):
		cf = theano.function(inputs = self.model.input,
				outputs=[self.model.activation]
				)

		loss = theano.function(inputs = [self.model.activation, self.model.Y],
				outputs=[self.cost_function]
				)
		#s = np.array([ [10, 2, 3], [4,5,6]])
		#y = np.array([1,2])
		#print loss(s, y) --> 11

		train_function = theano.function(
				inputs=self.model.input + self.model.output,
				outputs=self.cost_function,
				updates=self.sgs_updates + self.adagrad_lr_updates + self.param_updates)
		predict_function = theano.function(
				inputs=self.model.input + self.model.output,
				outputs=T.mean(T.eq(self.model.Y, self.model.predict))
				)

		for i in range(100):
			c = train_function(training_data[0], training_data[1], training_data[2])
			print self.model.b.get_value()
			print c
			accuracy = predict_function(training_data[0], training_data[1], training_data[2])
			print accuracy
