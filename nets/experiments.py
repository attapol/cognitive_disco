import sys
import timeit

import numpy as np
import scipy as sp
import theano.tensor as T

from cognitive_disco.data_reader import extract_implicit_relations
from bilinear_layer import BilinearLayer, LinearLayer
from learning import AdagradTrainer
import feature_functions as f
import base_label_functions as l
import util


def net_experiment0_0(dir_list):
	"""This setup should be deprecated"""
	brown_dict = util.BrownDictionary()
	relation_list_list = [util.convert_level2_labels(extract_implicit_relations(dir))
		for dir in dir_list]
	data_list, alphabet = brown_dict.get_brown_matrices_data(relation_list_list, False)
	num_features = data_list[0][0].shape[1]
	num_outputs = len(alphabet)

	rng = np.random.RandomState(12)
	blm = BilinearLayer(rng, num_features, num_features, num_outputs, activation_fn=None)
	trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
	trainer.train_minibatch(50, 20, data_list[0], data_list[1], data_list[2])

def net_experiment0_1(dir_list):
	"""Use Brown pair only"""
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

def net_experiment0_2(dir_list):
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

def net_experiment0_3(dir_list):
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



if __name__ == '__main__':
	experiment_name = sys.argv[1]
	if len(sys.argv) < 4:
		dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
		#dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
	else:
		dir_list = sys.argv[3:]
	globals()[experiment_name](dir_list)
	
