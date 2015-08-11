import sys

from cognitive_disco.data_reader import extract_implicit_relations
from bilinear_layer import BilinearLayer
from learning import AdagradTrainer
import numpy as np
import util

def net_experiment0(dir_list):
	pass

def net_experiment1(dir_list):
	brown_dict = util.BrownDictionary()
	relation_list_list = [util.convert_level2_labels(extract_implicit_relations(dir))
		for dir in dir_list]
	data_list, alphabet = util.get_brown_matrices_data(brown_dict, relation_list_list, False)
	num_features = data_list[0][0].shape[1]
	num_outputs = len(alphabet)

	rng = np.random.RandomState(12)
	blm = BilinearLayer(rng, num_features, num_features, num_outputs, activation_fn=None)
	trainer = AdagradTrainer(blm, blm.hinge_loss, 0.01, 0.01)
	trainer.train_minibatch(50, 20, data_list[0], data_list[1], data_list[2])


if __name__ == '__main__':
	experiment_name = sys.argv[1]
	if len(sys.argv) < 4:
		dir_list = ['conll15-st-05-19-15-train', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
		dir_list = ['conll15-st-05-19-15-dev', 'conll15-st-05-19-15-dev', 'conll15-st-05-19-15-test']
	else:
		dir_list = sys.argv[3:]
	globals()[experiment_name](dir_list)
	
