import numpy as np
import theano
from theano import config
import theano.tensor as T


def prep_srm_arg(relation_list, arg_pos, wbm, max_length):
	assert arg_pos == 1 or arg_pos == 2
	n_samples = len(relation_list)
	lengths = [min(max_length, len(x.arg_tokens(arg_pos))) for x in relation_list]
	x = np.zeros((max_length, n_samples)).astype('int64')
	x_mask = np.zeros((max_length, n_samples)).astype(config.floatX)
	for i, relation in enumerate(relation_list):
		indices = wbm.index_tokens(relation.arg_tokens(arg_pos))
		sequence_length = min(max_length, len(indices))
		x[:sequence_length, i] = indices[:sequence_length]
		x_mask[:sequence_length, i] = 1.
	embedding_series = wbm.wm[x.flatten()].reshape([max_length, n_samples, wbm.num_units]).astype(config.floatX)
	return embedding_series, x_mask

def prep_serrated_matrix_relations(relation_list, wbm, max_length):
	arg1_srm, arg1_mask = prep_srm_arg(relation_list, 1, wbm, max_length)
	arg2_srm, arg2_mask = prep_srm_arg(relation_list, 2, wbm, max_length)
	return (arg1_srm, arg1_mask, arg2_srm, arg2_mask)

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

class LSTM(object):

	def reset(self, rng):
		W_values = np.concatenate([self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj)], axis=1)
		U_values = np.concatenate([self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj)], axis=1)
		b_values = np.zeros((4 * self.dim_proj,)).astype(config.floatX)

		self.W.set_value(W_values)
		self.U.set_value(U_values)
		self.b.set_value(b_values)


	def __init__(self, rng, dim_proj, n_out=None, X=None, Y=None, activation_fn=None):
		self.params = []#a list of paramter variables
		self.input = [] #a list of input variables
		self.output = []#a list of output variables
		self.predict = [] # a list of prediction functions
		self.hinge_loss = None # a function
		self.crossentropy = None # a function

		self.dim_proj = dim_proj
		self.rng = rng

		#LSTM parameters
		W_values = np.concatenate([self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj)], axis=1)
		U_values = np.concatenate([self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj),
							self.square_weight(self.dim_proj)], axis=1)
		b_values = np.zeros((4 * self.dim_proj,)).astype(config.floatX)

		self.W = theano.shared(W_values, borrow=True)
		self.U = theano.shared(U_values, borrow=True)
		self.b = theano.shared(b_values, borrow=True)

		# Logistic regression parameters
		if Y is None:
			self.params = [self.W, self.U, self.b]
		else:
			U_pred_values = np.zeros((self.dim_proj, n_out)).astype(config.floatX)
			U_pred = theano.shared(U_pred_values, borrow=True)
			b_pred_values = np.zeros((n_out,)).astype(config.floatX)
			b_pred = theano.shared(b_pred_values, borrow=True)
			self.params = [self.W, self.U, self.b, U_pred, b_pred]

		X = T.tensor3('x', dtype=config.floatX) 
		mask = T.matrix('mask', dtype=config.floatX)
		self.input = [X, mask]

		n_samples = X.shape[1]

		self.h = self.project(X, mask)

		self.max_pooled_h = (self.h * mask[:, :, None]).max(axis=0) 
		self.sum_pooled_h = (self.h * mask[:, :, None]).sum(axis=0) 
		self.mean_pooled_h = self.sum_pooled_h / mask.sum(axis=0)[:, None]
		self.top_h = self.h[mask.sum(axis=0).astype('int64')-1,T.arange(n_samples), :]

		# connect to logistic regression
		if Y is None:
			self.output = []
		else:
			self.output = [Y]
			net = T.dot(self.mean_pooled_h, U_pred) + b_pred
			activation = net if activation_fn is None else activation_fn(net)
			self.predict = [activation.argmax(1)]
			smoother = 1e-6
			#if pred.dtype == 'float16':
				#smoother = 1e-6
			self.crossentropy = -T.log(activation[T.arange(Y.shape[0]), Y] + smoother).mean()
			self.hinge_loss = None #TODO: implement hinge loss as well

	def square_weight(self, ndim):
		return self.rng.uniform(
						low=-np.sqrt(6. / (4 * ndim)),
						high=np.sqrt(6. / (4 * ndim)),
						size=(ndim, ndim)
					).astype(config.floatX)
		#W = np.random.randn(ndim, ndim) * 0.0001
		#u, s, v = np.linalg.svd(W)
		#return u.astype(config.floatX)

	def project(self, embedding_series, mask):
		nsteps = embedding_series.shape[0]
		if embedding_series.ndim == 3:
			n_samples = embedding_series.shape[1]
		else:
			n_samples = 1

		def _slice(_x, n, dim):
			if _x.ndim == 3:
				return _x[:, :, n * dim:(n + 1) * dim]
			return _x[:, n * dim:(n + 1) * dim]

		def _step(m_, x_, h_, c_):
			preact = T.dot(h_, self.U)
			preact += x_

			i = T.nnet.sigmoid(_slice(preact, 0, self.dim_proj))
			f = T.nnet.sigmoid(_slice(preact, 1, self.dim_proj))
			o = T.nnet.sigmoid(_slice(preact, 2, self.dim_proj))
			c = T.tanh(_slice(preact, 3, self.dim_proj))

			c = f * c_ + i * c
			c = m_[:, None] * c + (1. - m_)[:, None] * c_

			h = o * T.tanh(c)
			h = m_[:, None] * h + (1. - m_)[:, None] * h_

			return h, c

		previous_state = T.dot(embedding_series, self.W) + self.b

		rval, updates = theano.scan(_step,
									sequences=[mask, previous_state],
									outputs_info=[T.alloc(np_floatX(0.),
															n_samples,
															self.dim_proj),
												T.alloc(np_floatX(0.),
															n_samples,
															self.dim_proj)],
									n_steps=nsteps)
		return rval[0]

