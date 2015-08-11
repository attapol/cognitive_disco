import numpy as np
import theano
import theano.tensor as T
import timeit

class Trainer(object):

	def train_minibatch(self, minibatch_size, n_epochs, training_data, dev_data, test_data):

		index = T.lscalar() # index to minibatch
		T_training_data = [theano.shared(x, borrow=True) for x in training_data]
		givens = {self.model.output[0] : 
				T_training_data[-1][index * minibatch_size: (index + 1) * minibatch_size]}
		for i, input_var in enumerate(self.model.input):
			givens[input_var] = \
					T_training_data[i][index * minibatch_size: (index + 1) * minibatch_size]


		self.train_function = theano.function(
				inputs=[index],
				outputs=self.cost_function,
				updates=self.sgs_updates + self.adagrad_lr_updates + self.param_updates,
				givens=givens
				)

		#cost = theano.function(inputs = [self.model.activation, self.model.Y],
				#outputs=self.cost_function)
		accuracy = T.mean(T.eq(self.model.Y, self.model.predict))
		self.eval_function = theano.function(inputs=self.model.input + self.model.output,
				outputs=[accuracy, self.cost_function]
				)

		patience = 5000
		patience_increase = 2 # wait this much longer when a new best is found
		improvement_threshold = 0.995

		n_train_batches = training_data[0].shape[0] / minibatch_size
		validation_frequency = min(n_train_batches, patience /2 )
	
		best_validation_loss = np.inf
		test_score = 0 

		done_looping = False
		epoch = 0
		while (epoch < n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in xrange(n_train_batches):
				iter = (epoch - 1) * n_train_batches  + minibatch_index
				start_time = timeit.default_timer()
				c = self.train_function(minibatch_index)
				end_time = timeit.default_timer()
				#print 'Iteration %s took % second(s)' % (iter, end_time - start_time)
				if (iter + 1) % validation_frequency == 0:
					accuracy, c  = self.eval_function(*dev_data)
					print 'DEV: Iteration %s : accuracy = %s ; cost =%s' % (iter, accuracy, c)
					accuracy, c  = self.eval_function(*test_data)
					print 'TEST: Iteration %s : accuracy = %s ; cost =%s' % (iter, accuracy, c)
					


class AdagradTrainer(Trainer):

	def __init__(self, model, cost_function, learning_rate, lr_smoother):
		self.model = model
		self.cost_function = cost_function 
		self.learning_rate = learning_rate
		self.lr_smoother = lr_smoother

		gparams = [T.grad(cost=cost_function, wrt=x) for x in self.model.params]
		adagrad_rates = [theano.shared(value=np.zeros(param.get_value().shape)) 
				for param in self.model.params]
		sum_gradient_squareds = [theano.shared(value=np.zeros(param.get_value().shape)) 
				for param in self.model.params]

		self.sgs_updates = [(sgs, sgs + T.square(gparam)) 
			for sgs, gparam in zip(sum_gradient_squareds, gparams)]

		self.adagrad_lr_updates = [(adagrad_rate, 
			adagrad_rate + learning_rate / (lr_smoother + T.sqrt(sum_gradient_squared)))
			for adagrad_rate, sum_gradient_squared in zip(adagrad_rates, sum_gradient_squareds)]

		self.param_updates = [(param, param - adagrad_rate * gparam) 
				for param, gparam, adagrad_rate in zip(self.model.params, gparams, adagrad_rates)]



class SGDTrainer(Trainer):

	def __init__(self, model, cost_function, learning_rate):
		self.model = model
		self.cost_function = cost_function 
		self.learning_rate = learning_rate

		gparams = [T.grad(cost=cost_function, wrt=x) for x in self.model.params]

