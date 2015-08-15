from tpl.language.lexical_structure import WordEmbeddingDict

import numpy as np

class EmbeddingFeaturizer(object):

	TEST_WORD_EMBEDDING_FILE = '../lib/lexicon/google_word_vector/GoogleNews-vectors-negative300_test.txt'
	WORD_EMBEDDING_FILE = '../lib/lexicon/google_word_vector/GoogleNews-vectors-negative300.txt'
	OOV_VALUE = 0

	def __init__(self, word_embedding_file=WORD_EMBEDDING_FILE):
		self.word_embedding_dict = WordEmbeddingDict(word_embedding_file)

	def create_arg_matrix(self, arg_tokens):
		num_tokens = len(arg_tokens)
		sentence_matrix = np.zeros((num_tokens, self.word_embedding_dict.num_units))
		for i in xrange(num_tokens):
			if arg_tokens[i] in self.word_embedding_dict:
				sentence_matrix[i, :] = self.word_embedding_dict[arg_tokens[i]]
			else:
				sentence_matrix[i, :] = EmbeddingFeaturizer.OOV_VALUE
		return sentence_matrix
	
	def additive_args(self, relation_list):
		num_relations = len(relation_list)
		num_units = self.word_embedding_dict.num_units
		
		arg1_matrix = np.array([self.create_arg_matrix(x.arg_tokens(1)).sum(0) 
			for x in relation_list])
		arg2_matrix = np.array([self.create_arg_matrix(x.arg_tokens(2)).sum(0) 
			for x in relation_list])
		return arg1_matrix, arg2_matrix 
	
