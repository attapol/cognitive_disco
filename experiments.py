import sys

import feature_functions as f
import label_functions as l

from naming_functions import doc_id_relation_id_nf
from feature_file_generator import generate_feature_files 


def experiment_test(mapping_file, dir_list):
	ff_list = [f.bag_of_words]
	lf = l.OriginalLabel()
	nf = doc_id_relation_id_nf
	generate_feature_files(dir_list, ff_list, lf, nf, 'experiment0')

def experiment_mapping_test(mapping_file, dir_list):
	ff_list = [f.bag_of_words]
	dimension_mapper = l.GenericMapping(mapping_file)
	nf = doc_id_relation_id_nf
	lf_list = dimension_mapper.get_all_label_functions()
	generate_feature_files(dir_list, ff_list, lf_list, nf, 'experiment_mapping_test')


def experiment0(mapping_file, dir_list):
	pass

if __name__ == '__main__':
	#execute the function named 'experiment_name'
	experiment_name = sys.argv[1]
	mapping_file = sys.argv[2]
	dir_list = sys.argv[3:]
	globals()[experiment_name](mapping_file, dir_list)


