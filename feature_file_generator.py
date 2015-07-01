from codecs import open as copen
from data_reader import extract_implicit_relations
from label_functions import OriginalLabel

def generate_feature_files(dir_list, ff_list, lf, nf, prefix):
	for dir in dir_list:
		relations = extract_implicit_relations(dir)
		new_prefix = '%s/%s' % (dir, prefix)
		if isinstance(lf, list):
			make_sparse_feature_files_for_all_labels(relations, ff_list, lf, nf, new_prefix)
		else:
			make_sparse_feature_file(relations, ff_list, lf, nf, new_prefix)

def make_sparse_feature_file(relations, ff_list, lf, nf, prefix):
	"""Make sparse feature files

	Args
		relations : a list of data_reader.DRelation objects
		ff_list : a list of feature functions
		lf : a label function
		nf : a naming function

	"""
	label_name = lf.label_name()
	file_name = '%s.%s.features' % (prefix, label_name)
	file = copen(file_name, mode='w', encoding='utf8')
	for relation in relations:
		feature_vector = []
		for ff in ff_list:
			feature_vector.extend(ff(relation))	
		label = lf.label(relation)
		if label is None:
			continue
		name = nf(relation)
		write_name_label_features(name, label, feature_vector, file)
	file.close()

def make_sparse_feature_files_for_all_labels(relations, ff_list, lf_list, nf, prefix):
	feature_vectors = []
	for relation in relations:
		feature_vector = []
		for ff in ff_list:
			feature_vector.extend(ff(relation))	
		feature_vectors.append(feature_vector)
	for lf in lf_list:
		label_name = lf.label_name()
		file_name = '%s.%s.features' % (prefix, label_name)
		file = copen(file_name, mode='w', encoding='utf8')
		for relation, feature_vector in zip(relations, feature_vectors):
			label = lf.label(relation)
			if label is None:
				continue
			name = nf(relation)
			write_name_label_features(name, label, feature_vector, file)
		file.close()

def write_name_label_features(name, label, feature_vector, file):
	if len(feature_vector) == 0: feature_vector.append('NONE')
	file.write('%s\t%s\t%s\n' % (name, label, ' '.join(feature_vector)))
