"""Label functions

Each function should take a data_reader.DRelation object as an argument 
and output a label string.

"""

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


