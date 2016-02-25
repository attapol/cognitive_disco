import codecs 
import itertools
import json
import sys
import random
import timeit

import numpy as np
import theano.tensor as T

import util
from tpl.pardo import pardo
import cognitive_disco.base_label_functions as l
from cognitive_disco.nets.bilinear_layer import make_multilayer_net
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.data_reader import DRelation

def extract_zh_implicit_relations(data_folder, label_function=None):
    parse_file = '%s/parses.json' % data_folder
    parse = json.load(codecs.open(parse_file, encoding='utf8'))

    relation_file = '%s/relations.json' % data_folder
    relation_dicts = [json.loads(x) for x in open(relation_file)]
    relations = [DRelation(x, parse) for x in relation_dicts if x['Type'] == 'Implicit']
    if label_function is not None:
        relations = [x for x in relations if label_function.label(x) is not None]
    return relations

def get_xvalidated_datatriplet(data_matrix_pair, label_vector, label_alphabet, 
        num_fold, fold_index):
    """Get the crossvalidated data at a given fold index
    """
    num_relations = len(label_vector)
    fold_size = round(num_relations / num_fold)

    dev_start_idx = fold_size * fold_index
    dev_end_idx = dev_start_idx + fold_size
    extract_dev = lambda x: x[dev_start_idx:dev_end_idx]

    if fold_index == (num_fold - 1):
        extract_test = lambda x: x[0:fold_size]
        extract_train = lambda x: x[fold_size:(fold_size * 2)]
    else:
        test_start_idx = dev_end_idx 
        test_end_idx = test_start_idx + fold_size
        extract_test = lambda x: x[test_start_idx:test_end_idx]
        extract_train = lambda x: \
                np.hstack(x[0:dev_start_idx], x[test_end_idx,:])

    dev_data = map(extract_dev, data_matrix_pair)
    dev_label_vector = extract_dev(label_vector)
    test_data = map(extract_test, data_matrix_pair)
    test_label_vector = extract_test(label_vector)
    train_data = map(extract_train, data_matrix_pair)
    train_label_vector = extract_train(label_vector)

    return DataTriplet([train_data, dev_data, test_data], 
            [[train_label_vector], [dev_label_vector], [test_label_vector]], 
            [label_alphabet])

def _make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    for i, input_var in enumerate(input_vec):
        givens[input_var] = T_training_data[i][start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

def zh_experiment0(dir_list, args):
    """Feedforward neural network

    We will experiment with all configurations of embedding that we have.

    Max, mean, and sum pooling
    
    """
    experiment_name = sys._getframe().f_code.co_name    
    sense_lf = l.OriginalLabel()

    train, dev = [extract_zh_implicit_relations(dir, sense_lf) for dir in dir_list[0:2]]
    relation_list = train + dev
    label_vectors, label_alphabet = util.label_vectorize([relation_list], sense_lf)
    label_vector = label_vectors[0]

    random.seed(10)
    random.shuffle(relation_list)

    vec_types = ['cbow', 'skipgram']
    num_unit_list = [50, 100, 150, 200, 250, 300]
    projection_list = ['sum_pool', 'max_pool', 'mean_pool', 'top']
    num_hidden_layer_list = [0, 1, 2]

    param_combos = itertools.product(
            vec_types, num_unit_list, projection_list, num_hidden_layer_list)
    pardo(_zh_experiment0_helper, param_combos, 10, exp_name=experiment_name,
            relation_list=relation_list, label_vector=label_vector, label_alphabet=label_alphabet)


def _zh_experiment0_helper(vec_type, num_units, projection, num_hidden_layers, 
        exp_name, relation_list, label_vector, label_alphabet):
    params = [vec_type, str(num_units), projection, str(num_hidden_layers)] 
    file_name = '%s-%s' % (experiment_name, '-'.join(params))
    json_file = util.set_logger(file_name)

    word2vec_ff = util._get_zh_word2vec_ff(num_units, vec_type, projection)
    data_matrix_pair = word2vec_ff(relation_list) 

    learning_rate = 0.001
    lr_smoother = 0.01

    num_folds = 7
    for fold_index in xrange(num_folds):
        data_triplet = get_xvalidated_datatriplet(data_matrix_pair, label_vector, label_alphabet,
                num_folds, fold_index)
        num_reps = 15
        num_hidden_units_list = [50, 200, 300, 400] 
        for num_hidden_units in num_hidden_units_list:
            rng = np.random.RandomState(100)
            X_list = [T.matrix(), T.matrix()]
            net, layers = make_multilayer_net(rng, 
                    n_in_list=data_triplet.input_dimensions(),
                    X_list=X_list, Y=T.lvector(), use_sparse=False,
                    num_hidden_layers=num_hidden_layers, 
                    num_hidden_units=num_hidden_units, 
                    num_output_units=data_triplet.output_dimensions()[0],
                    output_activation_fn=T.nnet.softmax,
                    dropout=False)
            trainer = AdagradTrainer(net, net.crossentropy, 
                    learning_rate, lr_smoother, data_triplet, _make_givens)
            for rep in xrange(num_reps):
                random_seed = rep
                rng = np.random.RandomState(random_seed)
                net.reset(rng)
                trainer.reset()
                minibatch_size = np.random.randint(20, 60)
                n_epochs = 50

                start_time = timeit.default_timer()
                best_iter, best_dev_acc, best_test_acc = \
                        trainer.train_minibatch_triplet(minibatch_size, n_epochs)
                end_time = timeit.default_timer()
                print 'Training process takes %s seconds' % (end_time - start_time)
                print 'Best iteration is %s;' % best_iter + \
                        'Best dev accuracy = %s' % best_dev_acc + \
                        'Test accuracy =%s' % best_test_acc

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    dir_list = ['conll16st-zh-01-08-2016-train', 'conll16st-zh-01-08-2016-dev', 'conll16st-zh-01-08-2016-test']
    globals()[experiment_name](dir_list, sys.argv[3:])
