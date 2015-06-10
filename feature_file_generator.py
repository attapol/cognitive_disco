from codecs import open as copen

def make_sparse_feature_file(relations, ff_list, lf, nf, prefix):
	"""Make sparse feature files

	Args
		relations : a list of data_reader.DRelation objects
		ff_list : a list of feature functions
		lf : a label function
		nf : a naming function

	"""
	label_name = lf.label_name()
	file_name = '%s_%s.features' % (prefix, label_name)
	file = copen(file_name, 'w', encoding='utf8')
	for relation in relations:
		feature_vector = []
		for ff in ff_list:
			feature_vector.extend(ff(relation))	
		label = lf.label(relation)
		name = nf(relation)
		write_name_label_features(name, label, feature_vector, file)
	file.close()

def write_name_label_features(name, label, feature_vector, file):
	file.write('%s\t%s\t%s\n' % (name, label, ' '.join(feature_vector)))
