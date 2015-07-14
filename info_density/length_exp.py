import sys

from naming_functions import doc_id_relation_id_nf
from feature_file_generator import generate_feature_files 
import info_density.feature_functions as f
import label_functions as lf

def lexperiment0_0(mapping_file, dir_list):
	experiment_name = 'lexperiment0.0'
	training_dir = dir_list[0]
	l = f.LengthFeaturizer(training_dir)
	ff_list = [l.length_char, l.length_char_diff, 
			l.length_centered_char, 
			l.length_word, l.length_word_diff,
			l.length_centered_word, ]
	dimension_mapper = lf.GenericMapping(mapping_file)
	lf_list = dimension_mapper.get_all_label_functions()
	lf_list.append(lf.OriginalLabel())
	nf = doc_id_relation_id_nf
	generate_feature_files(dir_list, ff_list, lf_list, nf, experiment_name)

if __name__ == '__main__':
	#execute the function named 'experiment_name'
	experiment_name = sys.argv[1]
	mapping_file = sys.argv[2]
	if len(sys.argv) < 4:
		dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
	else:
		dir_list = sys.argv[3:]
	globals()[experiment_name](mapping_file, dir_list)


