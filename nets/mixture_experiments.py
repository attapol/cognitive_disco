"""Mixture Experiments


Using the surface features in combination with the continuous features
induced by word embeddings.

"""
import json
import sys
import timeit

import cognitive_disco.base_label_functions as l
import cognitive_disco.nets.util as util
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.learning import AdagradTrainer, DataTriplet
from cognitive_disco.nets.bilinear_layer import \
        NeuralNet, MixtureOfExperts, make_multilayer_net

from theano import config
import theano.sparse
import theano.tensor as T
import numpy as np
from scipy.sparse import coo_matrix

def net_mixture_experiment1(dir_list, args):
    """Experiment 1 MOE 

    Read the sparse feature matrices from the files
    Use Mixture of Experts model
    """
    sparse_feature_file = args[0]
    embedding_size = int(args[1])
    cont_num_hidden_layers = int(args[2])
    sparse_num_hidden_layers = int(args[3])
    mixture_num_hidden_layers = int(args[4])
    proj_type = args[5]

    experiment_name = sys._getframe().f_code.co_name    
    json_file = util.set_logger('%s_%sunits_%sh_%sh_%sh_%s' % \
            (experiment_name,  embedding_size, 
                cont_num_hidden_layers,
                sparse_num_hidden_layers, 
                mixture_num_hidden_layers,
                proj_type))

    data_triplet = _load_continuous_sparse_features(dir_list, embedding_size, 
            sparse_feature_file, proj_type)
    num_hidden_unit_list = [50, 200, 300, 400, 500]
    num_reps = 20
    for num_hidden_unit in num_hidden_unit_list:
        _net_mixture_experiment_helper(experiment_name, 
                json_file, data_triplet, num_reps, 
                sparse_num_hidden_layers, cont_num_hidden_layers,
                True, mixture_num_hidden_layers, num_hidden_unit, proj_type)

def net_mixture_experiment2(dir_list, args):
    """Experiment 2 Linearly combining features

    No gating network. We simply concatenate the hidden layer of 
    the continuous features with the sparse feature vector.

    """
    sparse_feature_file = args[0]
    embedding_size = int(args[1])
    cont_num_hidden_layers = int(args[2])
    sparse_num_hidden_layers = int(args[3])
    mixture_num_hidden_layers = int(args[4])
    proj_type = args[5]

    experiment_name = sys._getframe().f_code.co_name    
    json_file = util.set_logger('%s_%sunits_%sh_%sh_%sh_%s' % \
            (experiment_name,  embedding_size, 
                cont_num_hidden_layers,
                sparse_num_hidden_layers, 
                mixture_num_hidden_layers,
                proj_type))

    data_triplet = _load_continuous_sparse_features(dir_list, embedding_size, 
            sparse_feature_file, proj_type)
    num_hidden_unit_list = [50, 200, 400, 500]
    num_reps = 12
    for num_hidden_unit in num_hidden_unit_list:
        _net_mixture_experiment_helper(experiment_name, 
                json_file, data_triplet, num_reps, 
                sparse_num_hidden_layers, cont_num_hidden_layers,
                False, mixture_num_hidden_layers, num_hidden_unit, proj_type)

def _load_continuous_sparse_features(dir_list, embedding_size,
        sparse_feature_file, proj_type):
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]

    num_features = 250000
    id_to_sfv = read_sparse_vectors(dir_list, sparse_feature_file)
    sfv_data_list = [get_sfv(relation_list, id_to_sfv, num_features) 
            for relation_list in relation_list_list]

    word2vec_ff = util._get_word2vec_ff(embedding_size, proj_type)
    word2vec_data_list = [word2vec_ff(relation_list) 
            for relation_list in relation_list_list]

    data_list = [[x] + y for x, y in zip(sfv_data_list, word2vec_data_list)]

    label_vector_triplet, label_alphabet = \
            util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vector_triplet], [label_alphabet])
    return data_triplet

def _net_mixture_experiment_helper(experiment_name, 
        json_file, data_triplet, num_reps, sparse_num_hidden_layers, 
        cont_num_hidden_layers, 
        use_moe, mixture_num_hidden_layers,
        num_hidden_units, proj_type):
    rng = np.random.RandomState(100)
    learning_rate = 0.001
    lr_smoother = 0.01

    if use_moe:
        output_activation_fn = T.nnet.softmax
        n_out = data_triplet.output_dimensions()[0]
    else:
        output_activation_fn = T.tanh
        n_out = num_hidden_units

    # the first one must be sparse
    X_list = [theano.sparse.csr_matrix(), T.matrix(), T.matrix()]
    sf_net, sf_layers = make_multilayer_net(rng, 
            n_in_list=data_triplet.input_dimensions()[0:1],
            X_list=X_list[0:1], Y=T.lvector(), use_sparse=True,
            num_hidden_layers=sparse_num_hidden_layers, 
            num_hidden_units=num_hidden_units, 
            num_output_units=n_out,
            output_activation_fn=output_activation_fn)
    word2vec_net, word2vec_layers = make_multilayer_net(rng, 
            n_in_list=data_triplet.input_dimensions()[1:],
            X_list=X_list[1:], Y=T.lvector(), use_sparse=False,
            num_hidden_layers=cont_num_hidden_layers, 
            num_hidden_units=num_hidden_units, 
            num_output_units=n_out,
            output_activation_fn=output_activation_fn)

    if use_moe:
        complete_net = MixtureOfExperts(rng,
                n_in_list=data_triplet.input_dimensions(),
                expert_list=[sf_net, word2vec_net],
                X_list=X_list, Y=T.lvector(),
                num_hidden_layers=mixture_num_hidden_layers, 
                num_hidden_units=num_hidden_units)
    else:
        mixture_net, mixture_layers = make_multilayer_net(rng,
                n_in_list=[sf_layers[-1].n_out, word2vec_layers[-1].n_out],
                X_list=[sf_layers[-1].activation, word2vec_layers[-1].activation],
                Y=T.lvector(),
                use_sparse=False,
                num_hidden_layers=mixture_num_hidden_layers,
                num_hidden_units=num_hidden_units,
                num_output_units=data_triplet.output_dimensions()[0])
        complete_net = NeuralNet(sf_layers + word2vec_layers + mixture_layers)
        complete_net.input = X_list

    trainer = AdagradTrainer(complete_net, complete_net.crossentropy, 
            learning_rate, lr_smoother, data_triplet, _make_givens)
    for rep in xrange(num_reps):
        random_seed = rep
        rng = np.random.RandomState(random_seed)
        complete_net.reset(rng)
        trainer.reset()

        minibatch_size = np.random.randint(20, 60)
        n_epochs = 50

        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs)
        end_time = timeit.default_timer()
        print end_time - start_time 
        print best_iter, best_dev_acc, best_test_acc
        result_dict = {
                'test accuracy': best_test_acc,
                'best dev accuracy': best_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'minibatch size': minibatch_size,
                'learning rate': learning_rate,
                'lr smoother': lr_smoother,
                'experiment name': experiment_name,
                'num hidden units': num_hidden_units,
                'sparse num hidden layers': sparse_num_hidden_layers,
                'continuous num hidden layers': cont_num_hidden_layers,
                'cost function': 'crossentropy',
                'projection' : proj_type,
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))


def _make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    for i, input_var in enumerate(input_vec):
        givens[input_var] = T_training_data[i][start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

def read_sparse_vectors(dir_list, sparse_feature_file):
    id_to_sfv = {}
    for dir in dir_list:
        file_name = '%s/%s' % (dir, sparse_feature_file)
        for line in open(file_name):
            split_line = line.strip().split()
            id_to_sfv[split_line[0]] = [int(x) for x in split_line[1:]]
    return id_to_sfv

def get_sfv(relation_list, id_to_sfv, num_features=250000):
    rows = []
    columns = []
    data = []
    for i, relation in enumerate(relation_list):
        relation_id = relation.doc_relation_id
        columns += id_to_sfv[relation_id]
        for j in xrange(len(id_to_sfv[relation_id])):
            rows.append(i)
            data.append(1)
    return coo_matrix(
            (data, (rows, columns)),
            shape=(len(relation_list), num_features), 
            dtype=config.floatX).tocsr()
