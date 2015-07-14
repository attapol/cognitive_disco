from data_reader import DRelation, extract_implicit_relations


class LMFeaturizer(object):

	def __init__(self, lm_file):
		pass

class LengthFeaturizer(object):
	"""Length-based features

	I want to try some stupid features that the robot guy from 
	Citadel recommends to me. It seems stupid but it's not
	so it's time to try it out and see what happens.
	"""

	def __init__(self, training_dir):
		"""We have to compute some sufficient statistics over the data

		"""
		self.relations = extract_implicit_relations(training_dir)
		self.mean_length_char, self.mean_length_word = self._compute_mean_length()
		self.cache_chars = {}
		self.cache_words = {}
	
	def _compute_mean_length(self):
		sum_length_char = 0.0
		sum_length_word = 0.0
		for relation in self.relations:
			tokens = relation.arg_tokens(1)
			for token in tokens:
				sum_length_char += len(token)
			sum_length_word += len(tokens)
			tokens = relation.arg_tokens(2)
			for token in tokens:
				sum_length_char += len(token)
			sum_length_word += len(tokens)
		num_relations = len(self.relations)
		return sum_length_char * 2 / num_relations, sum_length_word * 2 / num_relations	


	def _length_char_wrapper(self, relation):
		if relation.doc_relation_id not in self.cache_chars:
			tokens = relation.arg_tokens(1)
			arg1_length = 0.0
			for token in tokens:
				arg1_length += len(token)
			tokens = relation.arg_tokens(2)
			arg2_length = 0.0
			for token in tokens:
				arg2_length += len(token)
			self.cache_chars[relation.doc_relation_id] = arg1_length, arg2_length
		return self.cache_chars[relation.doc_relation_id]

	def length_char(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_char_wrapper(relation)
		return ['ARG1_LENGTH_CHAR:%s' % arg1_length,
				'ARG2_LENGTH_CHAR:%s' % arg2_length]

	def length_char_diff(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_char_wrapper(relation)
		diff = arg2_length - arg1_length
		return ['LENGTH_CHAR_DIFF:%s' % diff]

	def length_centered_char(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_char_wrapper(relation)
		return ['ARG1_CLENGTH_CHAR:%s' % (arg1_length - self.mean_length_char) ,
				'ARG2_CLENGTH_CHAR:%s' % (arg2_length - self.mean_length_char)]

	def _length_word_wrapper(self, relation):
		if relation.doc_relation_id not in self.cache_words:
			tokens = relation.arg_tokens(1)
			arg1_length = float(len(tokens))
			tokens = relation.arg_tokens(2)
			arg2_length = float(len(tokens))
			self.cache_words[relation.doc_relation_id] = arg1_length, arg2_length
		return self.cache_words[relation.doc_relation_id]

	def length_word(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_word_wrapper(relation)
		return ['ARG1_LENGTH_WORD:%s' % arg1_length,
				'ARG2_LENGTH_WORD:%s' % arg2_length]

	def length_word_diff(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_word_wrapper(relation)
		diff = arg2_length - arg1_length
		return ['LENGTH_WORD_DIFF:%s' % diff]

	def length_centered_word(self, relation):
		assert(isinstance(relation, DRelation))
		arg1_length, arg2_length = self._length_word_wrapper(relation)
		return ['ARG1_CLENGTH_WORD:%s' % (arg1_length - self.mean_length_word),
				'ARG2_CLENGTH_WORD:%s' % (arg2_length - self.mean_length_word)]

