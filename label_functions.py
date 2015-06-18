"""Label functions

Each function should take a data_reader.DRelation object as an argument 
and output a label string.

"""
import json
import os


class LabelFunction(object):
	
	def label_name(self):
		raise NotImplementedError("Subclasses should implement this!")

	def label(self, drelation):
		raise NotImplementedError("Subclasses should implement this!")

class OriginalLabel(LabelFunction):

	def label_name(self):
		return 'original_label'

	def label(self, drelation):
		senses = drelation.senses
		return senses[0]

class TopLevelLabel(LabelFunction):

	def label_name(self):
		return 'original_label'

	def label(self, drelation):
		senses = drelation.senses
		return senses[0].split('.')[0]


"""Generic Mapper from JSON file

The mapper file should be in this format
{
	"label1" : {"dimension1": l1, "dimension2": l2 ...}
	"label2" : {"dimension1": l1, "dimension2": l2 ...}
	...
	"labeln" : {...
	...
}

"""
class GenericMapping(object):

	def __init__(self, json_file):
		self.mapping = json.load(open(json_file))
		base_name = os.path.basename(json_file)
		mapping_name = os.path.splitext(base_name)[0]

		self.dimension_set = self.validate_dimension_set(self.mapping)
		self.dimension_to_lf = {}
		for dimension in self.dimension_set:
			self.dimension_to_lf[dimension] = GenericMapping.JSONLabel(self.mapping, mapping_name, dimension)

	def get_label_function(self, dimension):
		return self.dimension_to_lf[dimension]

	def get_all_label_functions(self):
		return [self.dimension_to_lf[x] for x in self.dimension_to_lf]

	def validate_dimension_set(self, mapping):
		dimension_set = set()
		for sense in mapping:
			if len(dimension_set) > 0:
				new_dimension_set = set(mapping[sense].keys())
				assert (len(new_dimension_set) > 0 )
				assert (new_dimension_set == dimension_set)
				dimension_set = new_dimension_set
			else:
				dimension_set = set(mapping[sense].keys())
		return dimension_set


	class JSONLabel(LabelFunction):

		def __init__(self, mapping, mapping_name, dimension):
			self.mapping = mapping
			self.mapping_name = mapping_name
			self.dimension = dimension

		def label_name(self):
			return '%s.%s' % (self.mapping_name, self.dimension)

		def label(self, drelation):
			senses = drelation.senses
			if senses[0] not in self.mapping:
				print '%s NOT FOUND! Skipping' % senses[0]
				return None
			else:
				return self.mapping[senses[0]][self.dimension]


""" read json object (mapping) from file
    (not sure if we should keep that here)
"""

def read_json(file):
        with open(file) as data_file:
                mapping = json.load(data_file)

        return mapping



""" basic operation (additive/non-additive/causal/non-causal/n.a.)
"""

def dimension_one(drelation, mapping):
        senses = drelation.senses
        dimensions = []

        for sense in senses:
                dimensions.append(mapping[sense]['basic'])

        return dimensions



""" order (arg1 before arg2 => forward
      or  arg2 before arg1  => backward
      or  n.a.)
"""

def dimension_two(drelation, mapping):
        senses = drelation.senses
        dimensions = []

        for sense in senses:
                dimensions.append(mapping[sense]['order'])

        return dimensions


""" semantic / pragmatic (sem/prag/n.a.)
"""

def dimension_three(drelation):
        senses = drelation.senses
        dimensions = []

        for sense in senses:
                dimensions.append(mapping[sense]['sem_prag'])

        return dimensions



""" polarity (pos/neg/n.a.)
"""

def dimension_four(drelation):
        senses = drelation.senses
        dimensions = []

        for sense in senses:
                dimensions.append(mapping[sense]['polarity'])

        return dimensions



