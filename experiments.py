import sys

import feature_functions as f
import label_functions as l

from data_reader import extract_implicit_relations
from naming_functions import doc_id_relation_id_nf
from feature_file_generator import make_sparse_feature_file

def generate_feature_files(dir_list, ff_list, lf, nf, prefix):
	for dir in dir_list:
		relations = extract_implicit_relations(dir)
		prefix = '%s/%s' % (dir, prefix)
		make_sparse_feature_file(relations, ff_list, lf, nf, prefix)

def experiment_test(dir_list):
	ff_list = [f.bag_of_words]
	lf = l.OriginalLabel
	nf = doc_id_relation_id_nf
	generate_feature_files(dir_list, ff_list, lf, nf, 'experiment0')

def hello(dir_list):
	print "hello"
	print dir_list

if __name__ == '__main__':
	#execute the function named 'experiment_name'
	experiment_name = sys.argv[1]
	dir_list = sys.argv[2:]
	globals()[experiment_name](dir_list)


