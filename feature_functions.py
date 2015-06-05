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
	feature_vector = []
	for arg1_token in relation.arg1_tokens:
		for arg2_token in relation.arg2_tokens:
			feature = 'WP_%s_%s' % (arg1_token, arg2_token)
			feature = re.sub(':','COLON', feature)
			feature_vector.append(feature)

	arg_tokens = relation.arg1_tokens
	arg_tokens.extend(relation.arg2_tokens)
	for arg_token in arg_tokens:
		feature = 'WORD_%s' % arg_token
		feature = re.sub(':','COLON', feature)
		feature_vector.append(feature)
	return feature_vector

