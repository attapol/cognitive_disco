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

