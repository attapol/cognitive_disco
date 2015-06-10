"""Label functions

Each function should take a data_reader.DRelation object as an argument 
and output a label string.

"""
import json


class LabelFunction(object):
	
	@classmethod
	def label_name(self):
		raise NotImplementedError("Subclasses should implement this!")

	@classmethod
	def label(self, drelation):
		raise NotImplementedError("Subclasses should implement this!")

class OriginalLabel(LabelFunction):

	@classmethod
	def label_name(self):
		return 'original_label'

	@classmethod
	def label(self, drelation):
		senses = drelation.senses
		return senses[0]



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



