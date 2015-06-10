"""Feature functions

Each function should take a data_reader.DRelation object as an argument,
and output a list of feature strings. 
If a function reuses some values from the object over and over,
the implementation should move to the methods not in the feature functions.

"""
import re

def bag_of_words(relation):
	"""Bag of words features

	: needs to be replaced with a string because 
	it will mess up Mallet feature vector converter
	"""
	feature_vector = []
	arg_tokens = relation.arg1_tokens
	arg_tokens.extend(relation.arg2_tokens)
	for arg_token in arg_tokens:
		feature = 'BOW_%s' % arg_token
		feature = re.sub(':','COLON', feature)
		feature_vector.append(feature)
	return feature_vector

def word_pairs(relation):
	"""Word pair features

	Bob is hungry. He wants a burger --> 
		Bob_He, Bob_wants, Bob_a, ... hungry_burger

	: needs to be replaced with a string because 
	it will mess up Mallet feature vector converter
	"""
	feature_vector = []
	for arg1_token in relation.arg1_tokens:
		for arg2_token in relation.arg2_tokens:
			feature = 'WP_%s_%s' % (arg1_token, arg2_token)
			feature = re.sub(':','COLON', feature)
			feature_vector.append(feature)
	return feature_vector

class BrownClusterFeaturizer(object):
	"""Brown Cluster-based featurizer

	We will only load the lexicon once and reuse it for all instances.
	Python goodness allows us to treat function as an object that is 
	still bound to another object

	lf = BrownClusterFeaturizer()
	lf.brown_pairs <--- this is a feature function that is bound to the lexicon

	"""
	def __init__(self):
		self.word_to_brown_mapping = {}
		brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
		self._load_brown_clusters('resources/%s' % brown_cluster_file_name)

	def _load_brown_clusters(self, path):
		try:
			lexicon_file = open(path)
			for line in lexicon_file:
				cluster_assn, word, _ = line.split('\t')
				self.word_to_brown_mapping[word] = cluster_assn	
		except:
			print 'fail to load brown cluster data'

	def get_cluster_assignment(self, word):
		if word in self.word_to_brown_mapping:
			return self.word_to_brown_mapping[word]
		else:
			return 'UNK'

	def brown_pairs(self, relation):
		"""Brown cluster pair features
		
		From the shared task, this feature won NLP People's choice award.
		People like using them because they are so easy to implement and
		work decently well.

		"""
		feature_vector = []
		for arg1_token in relation.arg1_tokens:
			for arg2_token in relation.arg2_tokens:
				arg1_assn = self.get_cluster_assignment(arg1_token)
				arg2_assn = self.get_cluster_assignment(arg2_token)
				feature = 'WP_%s_%s' % (arg1_assn, arg2_assn)
				feature = re.sub(':','COLON', feature)
				feature_vector.append(feature)
		return feature_vector

