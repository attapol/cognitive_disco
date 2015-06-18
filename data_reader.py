import json

class DRelation(object):
	"""Implicit discourse relation object

	The object is created from the CoNLL-json formatted data.
	The format can be a bit clunky to get certain information. 
	So convenient methods should be implemented here mostly to be used
	by the feature functions
	"""

	def __init__(self, relation_dict, parse):
		self.relation_dict = relation_dict
		self.parse = parse	

	@property
	def senses(self):
		return self.relation_dict['Sense']

	@property
	def arg1_tokens(self):
		return self._arg_tokens(1)

	@property
	def arg2_tokens(self):
		return self._arg_tokens(2)


	def arg_tree(self, arg_pos):
		"""Extract the tree for the

		Returns:
			1) tree string
			2) token indices (not address tuples) of that tree. 
		"""
		assert(arg_pos == 1 or arg_pos == 2)
		trees, sentence_indices = self.arg_trees(arg_pos)	
		if arg_pos == 1:
			tree = trees[-1]
			sentence_index = sentence_indices[-1]
		elif arg_pos == 2:
			tree = trees[0]
			sentence_index = sentence_indices[0]

		key = 'Arg%s' % arg_pos
		token_indices = [x[4] for x in self.relation_dict[key]['TokenList'] if x[3] == sentence_index]
		return tree, token_indices

	@property
	def doc_id(self):
		return self.relation_dict['DocID']

	@property
	def relation_id(self):
		return self.relation_dict['ID']

	def _arg_tokens(self, arg_pos):
		key = 'Arg%s' % arg_pos
		token_list = self.relation_dict[key]['TokenList']	
		return [self.parse[self.doc_id]['sentences'][x[3]]['words'][x[4]][0] for x in token_list]

	def arg_trees(self, arg_pos):
		key = 'Arg%s' % arg_pos
		token_list = self.relation_dict[key]['TokenList']	
		sentence_indices = set([x[3] for x in token_list])
		return [self.parse[self.doc_id]['sentences'][x]['parsetree'] for x in sentence_indices], list(sentence_indices)

	def __repr__(self):
		return self.relation_dict.__repr__()

	def __str__(self):
		return self.relation_dict.__str__()

def extract_implicit_relations(data_folder):
	parse_file = '%s/pdtb-parses.json' % data_folder
	parse = json.load(open(parse_file))

	relation_file = '%s/pdtb-data.json' % data_folder
	relation_dicts = [json.loads(x) for x in open(relation_file)]
	relations = [DRelation(x, parse) for x in relation_dicts if x['Type'] == 'Implicit']
	return relations

