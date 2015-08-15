import json
import sys
import timeit

import numpy as np
import scipy as sp
import theano.tensor as T

from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.bilinear_layer import BilinearLayer, LinearLayer
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
import cognitive_disco.nets.util as util
import cognitive_disco.dense_feature_functions as df
import cognitive_disco.feature_functions as f
import cognitive_disco.base_label_functions as l


def net_experiment0_0(dir_list, args):
	"""This setup should be deprecated"""
	brown_dict = util.BrownDictionary()
	relation_list_list = [util.convert_level2_labels(extract_implicit_relations(dir))
		for dir in dir_list]
	data_triplet, alphabet = brown_dict.get_brown_matrices_data(relation_list_list, False)
	num_features = data_triplet[0][0].shape[1]
	num_outputs = len(alphabet)

	rng = np.random.RandomState(12)
	blm = BilinearLayer(rng, num_features, num_features, num_outputs, activation_fn=None)
	trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
	trainer.train_minibatch(50, 20, data_triplet[0], data_triplet[1], data_triplet[2])

def net_experiment0_1(dir_list, args):
	"""Use Brown word and Brown pair only"""
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

	bf = f.BrownClusterFeaturizer()
	ff_list = [bf.brown_words, bf.brown_word_pairs]
	#ff_list = [f.modality]
	sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	rng = np.random.RandomState(12)

	lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
			use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
	trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
	start_time = timeit.default_timer()
	print trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
		[sfeature_matrices[1], label_vectors[1]],
		[sfeature_matrices[2], label_vectors[2]])
	end_time = timeit.default_timer()
	print end_time - start_time 

def net_experiment0_2(dir_list, args):
	"""Use feature selection"""
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

	ff_list = [f.production_rules]
	sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	mi = util.compute_mi(sfeature_matrices[0], label_vectors[0])
	sfeature_matrices = util.prune_feature_matrices(sfeature_matrices, mi, 500)

	rng = np.random.RandomState(12)
	lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
			use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
	trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
	start_time = timeit.default_timer()
	trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
		[sfeature_matrices[1], label_vectors[1]],
		[sfeature_matrices[2], label_vectors[2]])
	end_time = timeit.default_timer()
	print end_time - start_time 

def net_experiment0_3(dir_list, args):
	""" Use all surface features 
	"""
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
	bf = f.BrownClusterFeaturizer()
	plf = f.LexiconBasedFeaturizer()
	ff_list = [
			f.is_arg1_multiple_sentences, 
			f.first_last_first_3, f.average_vp_length, f.modality, 
			plf.inquirer_tag_feature, 
			plf.mpqa_score_feature, plf.levin_verbs, 
			f.production_rules, bf.brown_words, 
			bf.brown_word_pairs
			]
	sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	rng = np.random.RandomState(12)
	lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
			use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
	trainer = AdagradTrainer(lm, lm.crossentropy, 0.01, 0.01)
	start_time = timeit.default_timer()
	print trainer.train_minibatch(50, 20, [sfeature_matrices[0], label_vectors[0]],
		[sfeature_matrices[1], label_vectors[1]],
		[sfeature_matrices[2], label_vectors[2]])
	end_time = timeit.default_timer()
	print end_time - start_time 

# net_experiment1_x series
# Investigate the efficacy of bilinearity and feature abstraction through hidden layers
#

def set_logger(file_name):
	sys.stdout = open('%s.log' % file_name, 'w', 1)
	json_file = open('%s.json' % file_name, 'w', 1)
	return json_file

def _net_experiment1_sparse_helper(dir_list, experiment_name, ff_list):
	json_file = set_logger(experiment_name)
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]
	sfeature_matrices, alphabet = util.sparse_featurize(relation_list_list, ff_list)
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	for rep in xrange(15):
		random_seed = rep
		minibatch_size = 50
		n_epochs = 20
		learning_rate = 0.01
		lr_smoother = 0.01

		rng = np.random.RandomState(random_seed)
		lm = LinearLayer(rng, [sfeature_matrices[0].shape[1]], len(label_alphabet), 
				use_sparse=True, Y=T.lvector(), activation_fn=T.nnet.softmax)
		trainer = AdagradTrainer(lm, lm.crossentropy, learning_rate, lr_smoother)
		start_time = timeit.default_timer()
		best_iter, best_dev_acc, best_test_acc = \
				trainer.train_minibatch(minibatch_size, n_epochs, 
					[sfeature_matrices[0], label_vectors[0]],
					[sfeature_matrices[1], label_vectors[1]],
					[sfeature_matrices[2], label_vectors[2]])
		end_time = timeit.default_timer()
		print end_time - start_time 
		print best_iter, best_dev_acc, best_test_acc
		result_dict = {
				'test accuracy': best_test_acc,
				'best dev accuracy': best_dev_acc,
				'best iter': best_iter,
				'random seed': random_seed,
				'minibatch size': minibatch_size,
				'learning rate': learning_rate,
				'lr smoother': lr_smoother,
				'experiment name': experiment_name,
				}
		json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def net_experiment1_brown_l(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	bf = f.BrownClusterFeaturizer()
	ff_list = [bf.brown_words]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_brown_bl(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	bf = f.BrownClusterFeaturizer()
	ff_list = [bf.brown_word_pairs]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_brown_l_bl(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	bf = f.BrownClusterFeaturizer()
	ff_list = [bf.brown_words, bf.brown_word_pairs]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_l(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	ff_list = [f.production_singles]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_bl(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	ff_list = [f.production_pairs]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_production_l_bl(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	ff_list = [f.production_singles, f.production_pairs]
	_net_experiment1_sparse_helper(dir_list, experiment_name, ff_list)

def net_experiment1_word2vec_l(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	json_file = set_logger(experiment_name)
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

	ef = df.EmbeddingFeaturizer()
	data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
	
	for rep in xrange(30):
		random_seed = rep
		minibatch_size = 50
		n_epochs = 20
		learning_rate = 0.01
		lr_smoother = 0.01

		rng = np.random.RandomState(random_seed)
		lm = LinearLayer(rng, data_triplet.input_dimensions(), len(label_alphabet), 
				use_sparse=False, Y=T.lvector(), activation_fn=T.nnet.softmax)
		trainer = AdagradTrainer(lm, lm.crossentropy, learning_rate, lr_smoother)
		start_time = timeit.default_timer()
		best_iter, best_dev_acc, best_test_acc = \
				trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
		end_time = timeit.default_timer()
		print end_time - start_time 
		print best_iter, best_dev_acc, best_test_acc
		result_dict = {
				'test accuracy': best_test_acc,
				'best dev accuracy': best_dev_acc,
				'best iter': best_iter,
				'random seed': random_seed,
				'minibatch size': minibatch_size,
				'learning rate': learning_rate,
				'lr smoother': lr_smoother,
				'experiment name': experiment_name,
				}
		json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_bl(dir_list, args):
	experiment_name = sys._getframe().f_code.co_name	
	json_file = set_logger(experiment_name)
	lf = l.SecondLevelLabel()
	relation_list_list = [extract_implicit_relations(dir, lf) for dir in dir_list]

	ef = df.EmbeddingFeaturizer()
	data_list = [ef.additive_args(relation_list) for relation_list in relation_list_list]
	label_vectors, label_alphabet = util.label_vectorize(relation_list_list, lf)
	data_triplet = DataTriplet(data_list, [[x] for x in label_vectors])
	for rep in xrange(30):
		random_seed = rep
		minibatch_size = 50
		n_epochs = 20
		learning_rate = 0.01
		lr_smoother = 0.01

		rng = np.random.RandomState(random_seed)

		blm = BilinearLayer(rng, data_triplet.input_dimensions()[0],
			data_triplet.input_dimensions()[1], len(label_alphabet),
			Y=T.lvector(), activation_fn=T.nnet.softmax)

		trainer = AdagradTrainer(blm, blm.crossentropy, learning_rate, lr_smoother)
		start_time = timeit.default_timer()
		best_iter, best_dev_acc, best_test_acc = \
				trainer.train_minibatch_triplet(minibatch_size, n_epochs, data_triplet)
		end_time = timeit.default_timer()
		print end_time - start_time 
		print best_iter, best_dev_acc, best_test_acc
		result_dict = {
				'test accuracy': best_test_acc,
				'best dev accuracy': best_dev_acc,
				'best iter': best_iter,
				'random seed': random_seed,
				'minibatch size': minibatch_size,
				'learning rate': learning_rate,
				'lr smoother': lr_smoother,
				'experiment name': experiment_name,
				}
		json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))

def net_experiment1_word2vec_l_bl(dir_list, args):
	pass


if __name__ == '__main__':
	experiment_name = sys.argv[1]
	dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
	#dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
	globals()[experiment_name](dir_list, sys.argv[2:])
	
